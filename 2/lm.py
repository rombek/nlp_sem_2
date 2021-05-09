import re
import os
import torch
import numpy as np
from tqdm import trange, tqdm
from collections import defaultdict, Counter

START_TOKEN =  '<start>'
EOS_TOKEN =  '<eos>'

class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.U_input = torch.nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.BU_input = torch.nn.Parameter(torch.Tensor(4 * hidden_size))

        self.W_hidden = torch.nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.BW_hidden = torch.nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def forward(self, inp: torch.Tensor, cell_state: torch.Tensor, hidden_state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        hidden_mult = hidden_state @ self.W_hidden + self.BW_hidden
        input_mult  = inp @ self.U_input + self.BU_input 
        matr_sum = input_mult + hidden_mult

        f, i, c_new, o, = matr_sum.chunk(chunks=4, dim=1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        c_new = torch.tanh(c_new)
        o = torch.sigmoid(o)

        cell_state_new = (cell_state * f) + (i * c_new)
        hidden_state_new = o * torch.tanh(cell_state_new)

        return cell_state_new, hidden_state_new
        
    def reset_parameters(self):

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

class LSTMLayer(torch.nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.input_size = emb_size
        self.hidden_size = hidden_size
        self.LSTMCell = LSTMCell(emb_size, hidden_size)
        
    def forward(self, X_batch, initial_states):
        cell_state, hidden_state = initial_states
        outputs = []
        for timestamp in range(X_batch.shape[0]):
            cell_state, hidden_state = self.LSTMCell(X_batch[timestamp], cell_state, hidden_state)
            outputs.append(hidden_state)
        return torch.stack(outputs), (cell_state, hidden_state)

class LSTM(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, dropout_rate):
        super(LSTM, self).__init__()
        self.input_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.layers = []
        for i in range(num_layers):
            self.layers.append(torch.nn.Dropout(p=self.dropout_rate))
            if i == 0:
                self.layers.append(LSTMLayer(emb_size, hidden_size))
            else:
                self.layers.append(LSTMLayer(hidden_size, hidden_size))

        self.layers.append(torch.nn.Dropout(p=self.dropout_rate))    
        self.layers = torch.nn.ModuleList(self.layers)
            
    def forward(self, X_batch, initial_states):
        for ind, layer in enumerate(self.layers):
            if ind % 2 == 1:
                X_batch, states = layer(X_batch, initial_states)
            else:
                X_batch = layer(X_batch)
        return X_batch, states

class PTBLM(torch.nn.Module):
    def __init__(self, num_layers, emb_size, hidden_size, vocab_size, dropout_rate, weight_init=0.1, tie_emb=True):
        super(PTBLM, self).__init__()
        self.num_layers = num_layers
        self.input_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.weight_max = weight_init
        self.tie = tie_emb


        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.LSTM = LSTM(emb_size, hidden_size, num_layers, dropout_rate)
        self.decoder = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.tie_b =  torch.nn.Parameter(torch.zeros(vocab_size))

        self.init_weights()

        
    def forward(self, model_input, initial_states):
        embs = self.embedding(model_input).transpose(0, 1).contiguous()
        
        outputs, states = self.LSTM(embs, initial_states)
        
        # print(outputs.shape)
        if self.tie:
            ns, bs = outputs.shape[0], outputs.shape[1]
            outputs = outputs.view(-1, self.hidden_size)
            logits = outputs.mm(self.embedding.weight.t()) + self.tie_b
            logits = logits.view(ns, bs, self.vocab_size)
        else:
            logits = self.decoder(outputs)

        logits = logits.transpose(0, 1).contiguous()

        return logits, states
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-self.weight_max, self.weight_max)
        self.decoder.weight.data.uniform_(-self.weight_max, self.weight_max)
        torch.nn.init.uniform_(self.tie_b, -self.weight_max, self.weight_max)
        
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size).to(device), torch.zeros(batch_size, self.hidden_size).to(device)

def batch_generator_inds(data, word_to_id, batch_size, num_steps):
    L_tokens = data
    L_shifted = L_tokens[1:]
    L_tokens = L_tokens[:-1]
    
    slice_len = len(L_tokens) // batch_size
    X_lists = [L_tokens[i * slice_len : (i + 1) * slice_len] for i in range(batch_size)]
    Y_lists = [L_shifted[i * slice_len : (i + 1) * slice_len] for i in range(batch_size)] 
    total_batchs = slice_len // num_steps
    for i in range(total_batchs):
        X_batch = []
        Y_batch = []
        for lst in X_lists:
            X_batch.append(lst[i * num_steps : (i + 1) * num_steps])
        for lst in Y_lists:
            Y_batch.append(lst[i * num_steps : (i + 1) * num_steps])
        yield torch.tensor(X_batch, requires_grad=False), torch.tensor(Y_batch, requires_grad=False)

def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def run_epoch(
    lr,
    model,
    data, 
    word_to_id, 
    loss_fn, 
    batch_size,
    num_steps,
    optimizer = None, 
    clip_value=None,
    device = None
) -> float:
    '''
    Performs one training epoch or inference epoch
    Args:
        lr: Learning rate for this epoch
        model: Language model object
        data: Data that will be passed through the language model
        char_to_id: Mapping of each character into its index in the vocabulary
        loss_fn: Torch loss function
        optimizer: Torch optimizer
        device: Input tensors should be sent to this device
    Returns: 
        Perplexity
    '''
    
    total_loss, total_examples = 0.0, 0
    generator = batch_generator_inds(data, word_to_id=word_to_id, batch_size=batch_size, num_steps=num_steps)

    initial_state = model.init_hidden(batch_size=batch_size, device=device)
    for step, (X, Y) in enumerate(generator):
        X = X.to(device)
        Y = Y.to(device)
        
        logits, new_state = model(X, initial_state)
        initial_state = (new_state[0].detach(), new_state[1].detach())
        
        loss = loss_fn(logits.view((-1, model.vocab_size)), Y.view(-1))
        total_examples += loss.size(0)
        total_loss += loss.sum().item()
        loss = loss.mean()

        if optimizer is not None:
            # We have a new learning rate value at every step, so it needs to be updated
            update_lr(optimizer, lr)
            
            # Gradients computation
            loss.backward()
            
            # Gradient clipping by predefined norm value - usually 5.0
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Applying gradients - one gradient descent step
            optimizer.step()
            optimizer.zero_grad()

    return np.exp(total_loss / total_examples)

config = { 'batch_size': 20, 'num_steps': 35, 
           'num_layers': 2, 'emb_size': 1500,
           'hidden_size': 1500, 'vocab_size': -1,
           'dropout_rate': 0.65, 'num_epochs': 40,
           'learning_rate': 1.0, 'lr_decay' : 1.15,
           'epoch_decay' : 14, 'weight_init': 0.04, 
           'grad_clipping': 10,
           'optimizer':'Momentum'
         }

medium_config = { 'batch_size': 20, 'num_steps': 35, 
           'num_layers': 2, 'emb_size': 650,
           'hidden_size': 650, 'vocab_size': -1,
           'dropout_rate': 0.5, 'num_epochs': 25,
           'learning_rate': 1.0, 'lr_decay' : 0.9,
           'epoch_decay' : 10, 'weight_init': 0.05, 
           'grad_clipping':5,
           'optimizer':'Momentum'
         }



small_config = { 'batch_size': 64, 'num_steps': 35, 
           'num_layers': 2, 'emb_size': 256,
           'hidden_size': 256, 'vocab_size': -1,
           'dropout_rate': 0.2, 'num_epochs': 13,
           'learning_rate': 0.01, 'lr_decay' : 0.9,
           'epoch_decay' : 6, 'weight_init': 0.1, 
           'grad_clipping':5,
           'optimizer':'Adam'
         }

def train(token_list, word_to_id, id_to_word):
    """
    Trains n-gram language model on the given train set represented as a list of token ids.
    :param token_list: a list of token ids
    :return: learnt parameters, or any object you like (it will be passed to the next_proba_gen function)
    """



    ############################# REPLACE THIS WITH YOUR CODE #############################
    config['vocab_size'] = len(word_to_id)
    model = PTBLM(num_layers=config['num_layers'], emb_size=config['emb_size'],
              hidden_size=config['hidden_size'], vocab_size = len(word_to_id),
              dropout_rate=config['dropout_rate']
             )
    print("word_to_id: ", len(word_to_id))
    print(type(token_list), len(token_list))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.8)
    plot_data = []
    for i in trange(config['num_epochs']):
        lr_decay = config['lr_decay'] ** max(i + 1 - config['epoch_decay'], 0.0)
        if config['lr_decay'] > 1:
            lr_decay = 1 / lr_decay

        decayed_lr = config['learning_rate'] * lr_decay
        
        model.train()
        train_perplexity = run_epoch(decayed_lr, model, token_list, 
                                     word_to_id, loss_fn,
                                     config['batch_size'], config['num_steps'],
                                     optimizer=optimizer,
                                     clip_value=config['grad_clipping'],
                                     device=device)
    
        
        plot_data.append((i, train_perplexity, decayed_lr))
        tqdm.write(f'Epoch: {i+1}. Learning rate: {decayed_lr:.3f}. '
              f'Train Perplexity: {train_perplexity:.3f}. ' 
             )

    bs = config['batch_size']
    lr = config['learning_rate']
    lay = config['num_layers']
    save_path = f'./lstm-{bs}b-{lr}lr-{lay}layer-spec.pt'
    torch.save(model, save_path)
    return model, device
    ############################# REPLACE THIS WITH YOUR CODE #############################


def next_proba_gen(token_gen, params, hidden_state=None):
    """
    For each input token estimate next token probability distribution.
    :param token_gen: generator returning sequence of arrays of token ids (each array has batch_size independent ids);
     i-th element of next array is next token for i-th element of previous array
    :param params: parameters received from train function
    :param hidden_state: the initial state for next token that may be required 
     for sampling from the language model
    :param hidden_state: use this as the initial state for your language model(if it is not None).
     That may be required for sampling from the language model.

    :return: probs: for each array from token_gen should yield vector of shape (batch_size, vocab_size)
     representing predicted probabilities of each token in vocabulary to be next token.
     hidden_state: return the hidden state at each time step of your language model. 
     For sampling from language model it will be used as the initial state for the following tokens.
    """

    ############################# REPLACE THIS WITH YOUR CODE #############################

    model, device = params
    model.eval()

    with torch.no_grad():
        for token_arr in token_gen:
            X_batch = torch.tensor(token_arr.reshape(-1, 1)).to(device).long()
            
            if hidden_state is None:
                hidden_state = model.init_hidden(batch_size=X_batch.shape[0], device=device)        

            logits, new_state = model(X_batch, hidden_state)
            hidden_state = (new_state[0].detach(), new_state[1].detach())
            probs = torch.nn.functional.softmax(logits, -1).cpu().detach().numpy()[:,0,:]
            # print("PROBS shape = ", probs.shape)
            yield probs, hidden_state

    ############################# REPLACE THIS WITH YOUR CODE #############################

