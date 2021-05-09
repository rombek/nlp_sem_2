from collections import defaultdict, Counter
from itertools import chain
import os
import re
import string
import time
from typing import List, Any
import copy

import numpy as np
import scipy 

from tqdm import trange, tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.nn import functional as F


my_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this',
                  'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                  'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                  'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                  'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                  'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                  'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                  'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                  't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
                  've', 'y', 'ain', 'aren', "aren't", 'could', 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                  "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                  "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                  "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


class Preprocessor:
    def __init__(self):
        self.allowed_words = None
        self.w2ind = None
        self.ind2w = None
        self.special_tokens = ['<start>', '<eos>', '<pad>']

    def preproc_one_(self, text):
        text = text.lower()
        remove_tags = re.compile(r'<.*?>')
        text = re.sub(remove_tags, '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ''.join(sym if (sym.isalnum() or sym in (" ", "'")) else f" {sym} " for sym in text)
        return text
    
    def preproc_(self, texts):
        return [self.preproc_one_(text) for text in texts]
    
    def tokenize_one_(self, text, stem=0):
        """
            arg: list of texts
            return: list of tokenized texts
        """
    
        tokenizer = re.compile(r"-?\d*[.,]?\d+|[?'\w]+|\S", re.MULTILINE | re.IGNORECASE)
        tokenized_text = tokenizer.findall(text)
        if stem == 0:
            return [token for token in tokenized_text if token not in my_stop_words]
        stem_text = [token[:stem] for token in tokenized_text if token not in my_stop_words]
        return stem_text
    
    def tokenize_(self, texts):
        return [self.tokenize_one_(text) for text in texts]
    
    def make_vocab_(self, texts):
        data = [token for text in texts for token in text]
        data += self.special_tokens

        counter = Counter(data)
        sorted_words = sorted(counter.items(), key=lambda x: -x[1])
        words = [w for w, _ in sorted_words]
        self.w2ind = dict(zip(words, range(len(words))))
        self.ind2w = {v: k for k, v in self.w2ind.items()}

    def fit_vocab(self, texts, max_df = 0.5, min_df = 5, min_tf = 5):
        
        tmp_texts = self.preproc_(texts)
        tmp_texts = self.tokenize_(tmp_texts)
        
        self.allowed_words = set()
        
        df_cnt = defaultdict(int)
        tf_cnt = defaultdict(int)
        total_documents = len(tmp_texts)
        for text in tmp_texts:
            been = set()
            for token in text:
                if token not in been:
                    been.add(token)
                    df_cnt[token] += 1
                tf_cnt[token] += 1

        for word, tf in tf_cnt.items():
            df = df_cnt[word]
            if tf >= min_tf and df / total_documents <= max_df and df >= min_df:
                self.allowed_words.add(word)
            
        transformed_texts = self.transform_texts(tmp_texts, inside=True)
        self.make_vocab_(transformed_texts)
        return self
    
    def transform_texts(self, texts, inside=False):
        
        
        if self.allowed_words is None:
            raise RuntimeError("Need to fit before transform")
            
        if not inside:
            texts = self.preproc_(texts)
            texts = self.tokenize_(texts)
        
        new_texts = []
        for text in texts:
            new_text = []
            for token in text:
                if token in self.allowed_words:
                    new_text.append(token)
                else:
                    new_text.append('<unk>')
            new_texts.append(new_text)
        return new_texts

    def texts_to_inds(self, texts, max_len=None, mode='sent'):

        """
            Transform list of tokenized texts to torch tensors, ready for sentiment analysis.
            Return:
                dataset_inds: torch.tensor with texts indices
                text_lenghts: torch.tensor with lenght of each text, needed for more precise predicting.

        """

        if self.w2ind is None:
            raise RuntimeError("Need to fit vocab before transform")


        if mode == 'lm':
            inds_texts = []
            for text in texts:
                cur_text = []
                for token in text:
                    cur_text.append(self.w2ind[token])
                inds_texts.append(cur_text)
            return inds_texts

        if max_len is None:
            max_len = max(len(text) for text in texts)
        
        text_lenghts = np.array([min(len(text), max_len) - 1 for text in texts])
        dataset_inds = np.full(shape=(len(texts), max_len), fill_value=self.w2ind['<pad>'], dtype=np.int32)
        for text_ind, text in enumerate(texts):
            for token_ind, token in enumerate(text):
                if token_ind >= max_len:
                    break
                dataset_inds[text_ind, token_ind] = self.w2ind[token]

        return torch.LongTensor(dataset_inds), torch.tensor(text_lenghts)        


class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        '''
        Args:
            input_size: Size of token embedding
            hidden_size: Size of hidden state of LSTM cell
        '''
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Creating matrices whose weights will be trained
        # Token embedding (input of this cell) will be multiplied by this matrix
        self.U_input = torch.nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.BU_input = torch.nn.Parameter(torch.Tensor(4 * hidden_size))

        # Creating matrices whose weights will be trained
        # Hidden state from previous step will be multipied by this matrix
        # Zero hidden state at the initial step
        self.W_hidden = torch.nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.BW_hidden = torch.nn.Parameter(torch.Tensor(4 * hidden_size))

        # Weights initialization
        self.reset_parameters()

    def forward(self, inp: torch.Tensor, cell_state: torch.Tensor, hidden_state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Performes forward pass of the recurrent cell
        Args:
            inp: Output from Embedding layer at the current timestep
                Tensor shape is (batch_size, emb_size)
            cell_state: Output cell_state from previous recurrent step or zero state
                Tensor shape is (batch_size, hidden_size)
            hidden_state: Output hidden_state from previous recurrent step or zero state
                Tensor shape is (batch_size, hidden_size)
        Returns:
            Output from LSTM cell
        '''
        hidden_mult = hidden_state @ self.W_hidden + self.BW_hidden
        input_mult  = inp @ self.U_input + self.BU_input 
        matr_sum = input_mult + hidden_mult
        
        f, i, c_new, o, = matr_sum.chunk(chunks=4, dim=1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        c_new = torch.tanh(c_new)
        o = torch.sigmoid(o)
        
        cell_state_new = cell_state * f + i * c_new
        hidden_state_new = o * torch.tanh(cell_state_new)
        
        return cell_state_new, hidden_state_new
        
    def reset_parameters(self):
        '''
        Weights initialization
        '''
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
    def __init__(self, num_layers, emb_size, hidden_size, vocab_size, dropout_rate, weight_init=0.1, tie_emb=True, adaptive=False):
        super(PTBLM, self).__init__()
        self.num_layers = num_layers
        self.input_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.weight_max = weight_init
        self.tie = tie_emb
        self.adaptive = adaptive


        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.LSTM = LSTM(emb_size, hidden_size, num_layers, dropout_rate)
        self.decoder = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.tie_b =  torch.nn.Parameter(torch.zeros(vocab_size))

        self.adaptive_sm = torch.nn.AdaptiveLogSoftmaxWithLoss(self.hidden_size, self.vocab_size, cutoffs=[1000, 5000, 10000, 30000])
        
        self.sentiment_decoder = torch.nn.Linear(in_features=self.hidden_size, out_features=2)
        
        self.init_weights()

        
    def forward(self, model_input, initial_states, target=None):
        embs = self.embedding(model_input).transpose(0, 1).contiguous()
        
        outputs, states = self.LSTM(embs, initial_states)
        
        if self.adaptive:
            outputs = outputs.transpose(0, 1).contiguous()
            out, loss = self.adaptive_sm(outputs.view(-1, self.hidden_size), target.view(-1))
            return out, loss, states
        
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
    
    def forward_classify(self, batch_texts, text_lenghts, initial_states):
        """
            model_input: batch of indexed tests
            text_lenghts: lenght of examples in batch
            initial_states: states for lstm layers
        """
        embs = self.embedding(batch_texts).transpose(0, 1).contiguous()
        
        outputs, states = self.LSTM(embs, initial_states)
        # outputs.shape = (max_len, bs, hidden_size)
        
        max_len, bs = outputs.shape[0], outputs.shape[1]
        outputs = outputs.transpose(0, 1).contiguous()
        # outputs.shape = (bs, max_len, hidden_size)
        
        # Getting last non pad output vector
        outputs = outputs[np.arange(outputs.shape[0]), text_lenghts]
        #outputs.shape = (bs, hidden_size)
        
#         print(outputs.shape)
        
        logits = self.sentiment_decoder(outputs)
        
        return logits, states
    
    def predict(self, batch_texts, text_lenghts, initial_states):
        logits, states = self.forward_classify(batch_texts, text_lenghts, initial_states)
        _, predicted = torch.max(logits, 1)
        predicted = predicted.cpu()
        return predicted
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-self.weight_max, self.weight_max)
        self.decoder.weight.data.uniform_(-self.weight_max, self.weight_max)
        torch.nn.init.uniform_(self.tie_b, -self.weight_max, self.weight_max)
        
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size).to(device), torch.zeros(batch_size, self.hidden_size).to(device)


def print_batch(i, X_batch, Y_batch):
    print(f"batch # {i}")
    for i in range(len(X_batch)):
        print(X_batch[i], Y_batch[i])

def batch_generator_inds(texts, word_to_id, batch_size, num_steps):
    L_tokens = list(chain(*texts))
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
        #assert(len(X_batch) == batch_size)
        #assert(len(Y_batch) == batch_size)
        #assert(all(len(i) == num_steps for i in X_batch))
        #assert(all(len(i) == num_steps for i in Y_batch))
        X_b_tensor = torch.tensor(X_batch, requires_grad=False)
        Y_b_tensor = torch.tensor(Y_batch, requires_grad=False)
        yield X_b_tensor, Y_b_tensor

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
    clip_value = None, 
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
        
        if model.adaptive:
            out, loss, new_state = model(X, initial_state, target=Y)
        else:
            logits, new_state = model(X, initial_state)
        initial_state = (new_state[0].detach(), new_state[1].detach())
               
        if model.adaptive:
            total_examples += out.shape[0]
            total_loss += loss.item() * out.shape[0]
        else:
            loss = loss_fn(logits.view((-1, model.vocab_size)), Y.view(-1))
            total_examples += loss.size(0)
            total_loss += loss.sum().item()
            loss = loss.mean()
        
        
        # Gradients computation
        if optimizer is not None:
            loss.backward()

            # We have a new learning rate value at every step, so it needs to be updated
            update_lr(optimizer, lr)

            # Gradient clipping by predefined norm value - usually 5.0
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Applying gradients - one gradient descent step                    
            optimizer.step()
            optimizer.zero_grad()
            
    return np.exp(total_loss / total_examples)


sent_config = {
        'batch_size' : 256,
        'vocab_size': -1,
        'dropout_rate' : 0.65, 'num_epochs' : 5,
        'learning_rate': 0.0005, 'lr_decay' : 0.5,                    
        'epoch_decay' : 3, 'grad_clipping' : 5,
        'optimizer' : 'Adam'
        }

def sent_run_epoch(
    lr,
    model,
    loss_fn, 
    batch_size,
    dataloader,
    optimizer = None, 
    clip_value = None, 
    device = None
) -> float:
    '''
    Performs one training epoch or inference epoch
    Args:
        lr: Learning rate for this epoch
        model: Language model object
        dataloader: pytorch Dataloader with (text, label, len) examples
        word_to_id: Mapping of each word into its index in the vocabulary
        loss_fn: Torch loss function
        optimizer: Torch optimizer
        device: Input tensors should be sent to this device
    Returns: 
        Accuracy
    '''
    total_loss, total_examples = 0.0, 0
    total_correct = 0
    
    
    for step, (X_batch, Y_batch, len_batch) in enumerate(dataloader):

        initial_state = model.init_hidden(batch_size=X_batch.shape[0], device=device)
        
        X = X_batch.to(device)
        Y = Y_batch.to(device)
        lenghts = len_batch.to(device)
        
        logits, new_state = model.forward_classify(X, lenghts, initial_state)
                
        loss = loss_fn(logits, Y.view(-1))
        total_examples += loss.size(0)
        total_loss += loss.sum().item()
        loss = loss.mean()
        
        _, predicted = torch.max(logits, 1)
        predicted = predicted.cpu()
        total_correct += (Y_batch == predicted).sum().item()
        
        # Gradients computation
        if optimizer is not None:
            loss.backward()

            # We have a new learning rate value at every step, so it needs to be updated
            update_lr(optimizer, lr)

            # Gradient clipping by predefined norm value - usually 5.0
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Applying gradients - one gradient descent step                    
            optimizer.step()
            optimizer.zero_grad()
        
    return total_loss / total_examples, total_correct / total_examples

def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    """c
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :param pretrain_params: parameters that were learned at the pretrain step
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    # ############################ REPLACE THIS WITH YOUR CODE #############################
    print(len(train_texts))

    model, preproc = pretrain_params
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sent_config['vocab_size'] = len(preproc.w2ind)

    print("START TRAIN PREPROCESSING")
    start = time.time()
    preprocessed_texts = preproc.transform_texts(train_texts)

    print(f"FINISH TRAIN PREPROCESSING in {time.time() - start} sec.")

    y_train = np.array([int(lab == 'pos') for lab in train_labels]) 
    y_train = torch.tensor(y_train)

    X_train, train_text_lenghts = preproc.texts_to_inds(preprocessed_texts, max_len=200)
    print(X_train.shape)
    print(X_train.shape, y_train.shape, train_text_lenghts.shape)
    train_dataset = TensorDataset(X_train, y_train, train_text_lenghts)


    dev_size = int(0.15 * X_train.shape[0])
    train_size = X_train.shape[0] - dev_size
    train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset, [train_size, dev_size])

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    if sent_config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=sent_config['learning_rate'])
    elif sent_config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=sent_config['learning_rate'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=sent_config['learning_rate'], momentum=0.8)


    for i in trange(sent_config['num_epochs']):
        
        train_dataloader = DataLoader(train_dataset, batch_size=sent_config['batch_size'], shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=sent_config['batch_size'], shuffle=True)
        
        lr_decay = sent_config['lr_decay'] ** max(i + 1 - sent_config['epoch_decay'], 0.0)
        if sent_config['lr_decay'] > 1:
            lr_decay = 1 / lr_decay
        decayed_lr = sent_config['learning_rate'] * lr_decay
        model.train()
        train_loss, train_acc = sent_run_epoch(decayed_lr, model,  
                                     loss_fn,
                                     dataloader=train_dataloader,
                                     batch_size=sent_config['batch_size'],
                                     optimizer=optimizer,
                                     clip_value=sent_config['grad_clipping'],
                                     device=device)

        model.eval()
        with torch.no_grad():
            dev_loss, dev_acc =  sent_run_epoch(decayed_lr, model,  
                                     loss_fn,
                                     dataloader=dev_dataloader,
                                     batch_size=sent_config['batch_size'],
                                     clip_value=sent_config['grad_clipping'],
                                     device=device)

        print(f'Epoch: {i+1}. Learning rate: {decayed_lr}. '
              f'Train Acc: {train_acc:.3f}. '
              f'Train Loss: {train_loss:.3f}. '
              f'Dev Acc: {dev_acc:.3f}. '
              f'Dev Loss: {dev_loss:.3f}. '
             )
    
    model.train()

    print("Uplearning on dev")
    st = time.time()
    dev_dataloader = DataLoader(dev_dataset, batch_size=sent_config['batch_size'], shuffle=True)
    tmp_loss, tmp_acc = sent_run_epoch(sent_config['learning_rate'], model,  
                                     loss_fn,
                                     dataloader=dev_dataloader,
                                     batch_size=sent_config['batch_size'],
                                     optimizer=optimizer,
                                     clip_value=sent_config['grad_clipping'],
                                     device=device)
    print(f"Upleared in {time.time() - st} sec.")
    model.eval()

    return model, preproc
    # ############################ REPLACE THIS WITH YOUR CODE #############################


lm_config_momentum = { 
        'batch_size': 256, 'num_steps': 35, 
        'num_layers': 2, 'emb_size': 650,
        'hidden_size': 650, 'vocab_size': -1,
        'dropout_rate': 0.65, 'num_epochs': 5,
        'learning_rate': 1.0, 'lr_decay' : 0.8,
        'epoch_decay' : 10, 'tied_embs': False,
        'weight_init': 0.05, 'grad_clipping' : 5,
        'optimizer' : 'Momentum', 
        'adaptive' : True
        }

lm_config = { 
        'batch_size': 256, 'num_steps': 35, 
        'num_layers': 20, 'emb_size': 650,
        'hidden_size': 650, 'vocab_size': -1,
        'dropout_rate': 0.65, 'num_epochs': 10,
        'learning_rate': 0.005, 'lr_decay' : 0.8,
        'epoch_decay' : 8, 'tied_embs': False,
        'weight_init': 0.05, 'grad_clipping' : 5,
        'optimizer' : 'Adam', 
        'adaptive' : True
         }

def pretrain_texts_split(all_texts, split_ratio=[0.8, 0.2]):

    if sum(split_ratio) != 1:
        raise RuntimeError("Sum of split ratios must be 1.")

    parts = [[] for _ in range(len(split_ratio))]
    parts_len = [int(len(all_texts) * sr) for sr in split_ratio]

    delta = len(all_texts) - sum(parts_len)
    parts_len[-1] += delta


    indices = np.arange(len(all_texts))
    np.random.shuffle(indices)

    cur_part = 0
    cur_len = 0
    i = 0
    while cur_part < len(split_ratio):
        parts[cur_part].append(all_texts[indices[i]])
        i += 1
        cur_len += 1
        if cur_len == parts_len[cur_part]:
            cur_part += 1
            cur_len = 0

    return parts


def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    # ############################ PUT YOUR CODE HERE #######################################
    all_texts = list(chain(*texts_list))
    print(f"TOTAL TEXTS: {len(all_texts)}")
    print("START PREPROCESSING")
    start = time.time()
    preproc = Preprocessor()
    preproc = preproc.fit_vocab(all_texts, min_tf=5, min_df=5)
    preprocessed_texts = preproc.transform_texts(all_texts)

    preprocessed_texts_inds = preproc.texts_to_inds(preprocessed_texts, mode='lm')
#    print(preprocessed_texts_inds[0])
    print(f"FINISH PREPROCESSING in {time.time() - start} sec.")
    print("VOCAB_SIZE: ", len(preproc.w2ind))
    
    lm_config['vocab_size'] = len(preproc.w2ind)
            
    model = PTBLM(num_layers=lm_config['num_layers'], emb_size=lm_config['emb_size'],
              hidden_size=lm_config['hidden_size'], vocab_size=lm_config['vocab_size'],
              dropout_rate=lm_config['dropout_rate'], weight_init=lm_config['weight_init'],
              tie_emb=lm_config['tied_embs'], adaptive=lm_config['adaptive']
             )

    print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on device: ", device)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    if lm_config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lm_config['learning_rate'])
    elif lm_config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lm_config['learning_rate'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lm_config['learning_rate'], momentum=0.8)


    for i in trange(lm_config['num_epochs']):
        train_preprocessed_texts_inds, dev_preprocessed_texts_inds = pretrain_texts_split(preprocessed_texts_inds, [0.85, 0.15])

        #print(len(train_preprocessed_texts), len(dev_preprocessed_texts))


        lr_decay = lm_config['lr_decay'] ** max(i + 1 - lm_config['epoch_decay'], 0.0)
        if lm_config['lr_decay'] > 1:
            lr_decay = 1 / lr_decay
        decayed_lr = lm_config['learning_rate'] * lr_decay

        model.train()
        train_perplexity = run_epoch(decayed_lr, model, train_preprocessed_texts_inds,
                                     preproc.w2ind, loss_fn,
                                     lm_config['batch_size'], lm_config['num_steps'],
                                     optimizer=optimizer,
                                     clip_value=lm_config['grad_clipping'],
                                     device=device)

        
        with torch.no_grad():
            dev_perplexity =  run_epoch(decayed_lr, model, dev_preprocessed_texts_inds,
                                     preproc.w2ind, loss_fn,
                                     lm_config['batch_size'], lm_config['num_steps'],
                                     clip_value=lm_config['grad_clipping'],
                                     device=device)

    
        print(f'Epoch: {i+1}. Learning rate: {decayed_lr:.5f}. '
              f'Train Perplexity: {train_perplexity:.3f}. '
              f'Dev Perplexity: {dev_perplexity:.3f}. '
             )
    return model, preproc


def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
       
    # ############################ REPLACE THIS WITH YOUR CODE #############################
    model, preproc = params

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("START PREPROCESSING")
    start = time.time()
    preprocessed_texts = preproc.transform_texts(texts)

    print(f"FINISH PREPROCESSING in {time.time() - start} sec.")


    X_test, test_text_lenghts = preproc.texts_to_inds(preprocessed_texts, max_len=200)
#    print(X_test.shape)

    test_dataset = TensorDataset(X_test, test_text_lenghts)
    test_dataloader = DataLoader(test_dataset, batch_size=sent_config['batch_size'])

    predicts = []
    for X_batch, len_batch in test_dataloader:
        initial_state = model.init_hidden(batch_size=X_batch.shape[0], device=device)
        
        X = X_batch.to(device)
        lenghts = len_batch.to(device)
        
        predicted_batch = model.predict(X, lenghts, initial_state)
         
        predicts.append(predicted_batch)

    labels = ['pos' if lab == 1 else 'neg' for lab in chain(*predicts)]
    return labels
    # ############################ REPLACE THIS WITH YOUR CODE #############################
