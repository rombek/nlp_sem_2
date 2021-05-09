import time
from typing import List, Any
from random import random

import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import defaultdict
from functools import partial
from itertools import chain
import re
import string
from tqdm import tqdm


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

def preprocessing(text):
    text = text.lower()
    remove_tags = re.compile(r'<.*?>')
    text = re.sub(remove_tags, '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(sym if (sym.isalnum() or sym in (" ", "'")) else f" {sym} " for sym in text)
    return text


def tokenize_text(text, stem=0):
    """
        arg: list of texts
        return: list of tokenized texts
    """
    
    tokenizer = re.compile(r"-?\d*[.,]?\d+|[?'\w]+|\S", re.MULTILINE | re.IGNORECASE)
    tokenized_text = tokenizer.findall(text)
    if stem == 0:
        return [token for token in tokenized_text if token not in my_stop_words]
    stem_dataset = [token[:stem] for token in tokenized_text if token not in my_stop_words]
    return stem_dataset


def preprocess_texts(dataset_texts):
    processed_texts = []
    for ind, text in enumerate(dataset_texts):
        prepared_text = preprocessing(text)
        tokenized_text = tokenize_text(prepared_text)
        processed_texts.append(tokenized_text)
    return processed_texts


def generate_ngrams(text, max_ngram=3):
    ngrams = []
    for token in text:
        ngrams.append(token)
    if max_ngram >= 2:
        for token in zip(text[:-1], text[1:]):
            ngrams.append(' '.join(token))
    if max_ngram >= 3:
        for token in zip(text[:-2], text[1:-1], text[2:]):
            ngrams.append(' '.join(token))
    return ngrams


def make_ngram_dataset(dataset, max_ngram=3):
    ngram_dataset = []
    for ind, text in enumerate(dataset):
        ngrams = generate_ngrams(text)
        ngram_dataset.append(ngrams)
    return ngram_dataset


def make_vocab(texts, max_df=0.5, min_df=10, min_tf=3, max_tokens=1000000):
    print("MAKING VOCAB")
    start = time.time()
    df_cnt = defaultdict(int)
    tf_cnt = defaultdict(int)
    total_documents = len(texts)
    # print(f"total_documents = {total_documents}")
    for text in texts:
        been = set()
        for token in text:
            if token not in been:
                been.add(token)
                df_cnt[token] += 1
            tf_cnt[token] += 1

    free_ind = 0
    w2ind = dict()
    vocab_tf = []
    tf_with_inds = []
    for word, tf in tf_cnt.items():
        df = df_cnt[word]
        if tf >= min_tf and df / total_documents <= max_df and df >= min_df:
            w2ind[word] = free_ind
            vocab_tf.append(tf)
            tf_with_inds.append((tf, word))
            free_ind += 1

    tf_with_inds.sort(key=lambda x: x[0], reverse=True)
    for tf, w in tf_with_inds[max_tokens:]:
        del w2ind[w]

    free_ind = 0
    w2ind_final = dict()
    vocab_tf_final = []
    for w, ind in w2ind.items():
        w2ind_final[w] = free_ind
        vocab_tf_final.append(tf_cnt[w])
        free_ind += 1

    vocab_tf_final = np.array(vocab_tf_final, dtype=np.float64)
    vocab_tf_prob = np.float_power(vocab_tf_final, 0.75)
    vocab_tf_prob /= vocab_tf_prob.sum()


    print(f"Finish vocab in {time.time() - start} seconds.")
    return w2ind_final, vocab_tf_prob


def make_inds_ngram_dataset(texts, w2ind, shuffle=True):
    ngrams_inds = []
    docs_inds = []
    for doc_ind, text in enumerate(texts):
        for ngram in text:
            if ngram in w2ind:
                ngrams_inds.append(w2ind[ngram])
                docs_inds.append(doc_ind)

    ngrams_inds = np.array(ngrams_inds)
    docs_inds = np.array(docs_inds)
    assert (len(ngrams_inds) == len(docs_inds))
    if shuffle:
        permutation = np.random.permutation(len(docs_inds))
        ngrams_inds = ngrams_inds[permutation]
        docs_inds = docs_inds[permutation]
    return ngrams_inds, docs_inds


def batch_generator(words_idxs, docs_idxs, probs, nb=5, batch_size=100):
    # Let's generate all negative examples at once.

    neg_samples = np.random.choice(np.arange(len(probs)), size=(nb * len(words_idxs),), p=probs)

    # print("pos_samples_len = ", len(docs_idxs), ", neg_samples = ", len(neg_samples))

    end = (len(words_idxs) // batch_size - 1) * batch_size + 1
    for batch_start in range(0, end, batch_size):
        pos_batch = words_idxs[batch_start: batch_start + batch_size]
        neg_batches = neg_samples[batch_start + batch_size : batch_start + batch_size * (nb + 1)]        

        # print(type(pos_batch), type(neg_batches))
        # print(pos_batch.shape, neg_batches.shape)
        batches = np.concatenate((pos_batch, neg_batches))

        docs_batch = np.tile(docs_idxs[batch_start: batch_start + batch_size], (nb + 1))
        labels = np.array([1 for _ in range(len(pos_batch))] + [0 for _ in range(len(neg_batches))])

        perm = np.random.permutation(batch_size * (nb + 1))

        # print(len(batches), len(docs_batch), len(labels))
        for i in range(0, perm.shape[0], batch_size):
            inds = perm[i : i + batch_size]
            yield (batches[inds], docs_batch[inds], labels[inds])


def count_labels(labels: List):
    return {
        unique_label: sum(1 for label in labels if label == unique_label)
        for unique_label in set(labels)
    }


class Doc2Vec:
    def __init__(self, vocab_size, docs_cnt, emb_size=500, train_start=0):
        self.word_embs = np.random.uniform(low=-0.001, high=0.001, size=(vocab_size, emb_size))
        self.docs_embs = np.random.uniform(low=-0.001, high=0.001, size=(docs_cnt, emb_size))
        self.vocab_size = vocab_size
        self.docs_cnt = docs_cnt
        self.emb_size = emb_size
        self.train_start = train_start
        self.bow = None

    def train(self, word_inds, doc_inds, labels, lr):
        word_batch_embs = self.word_embs[word_inds]
        doc_batch_embs = self.docs_embs[doc_inds]

        dot_prods = np.einsum('ij,ij->i', word_batch_embs, doc_batch_embs)
        y_pred = self.sigmoid(dot_prods)

        word_batch_grads = doc_batch_embs * (y_pred - labels).reshape(-1, 1)
        doc_batch_grads = word_batch_embs * (y_pred - labels).reshape(-1, 1)

        for ind, (w_ind, d_ind) in enumerate(zip(word_inds, doc_inds)):
            self.word_embs[w_ind] -= lr * word_batch_grads[ind]
            self.docs_embs[d_ind] -= lr * doc_batch_grads[ind]

        batch_loss = (-labels * np.log(y_pred) - (1 - labels) * np.log(1 - y_pred)).sum()

        return batch_loss

    def get_X(self, start, X_len, mode='d2v'):
        """
            Returns vectors for LogisticRegression
            Work in two modes: d2v and d2vNB
        """
        if mode == 'd2v':
            return self.docs_embs[start : start + X_len]
        elif mode == 'd2vNB':
            full_embs = scipy.sparse.hstack((self.docs_embs, self.bow))
            full_embs = full_embs.tocsr()
            return full_embs[start : start + X_len]

    def set_bow_vectors(self, X_bow_full):
        self.bow = X_bow_full

    def apply_nb_weights(self, nb_weights):
        self.bow = (self.bow > 0) * scipy.sparse.diags(nb_weights)

    def get_X_bow(self, start, X_len):
        return self.bow[start : start + X_len]

    def sigmoid(self, x):
        return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.0))


def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :param pretrain_params: parameters that were learned at the pretrain step
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    # ############################ REPLACE THIS WITH YOUR CODE #############################


    doc2vec, datasets_info = pretrain_params

    X_train_start, X_train_len = datasets_info[train_texts[0]]

    y_train = np.array([int(lab == 'pos') for lab in train_labels])

    #------------------------------Fitting naive bayes started------------------------------
    X_train_bow = doc2vec.get_X_bow(X_train_start, X_train_len)
    clf_bayes = MultinomialNB()
    clf_bayes.fit(X_train_bow, y_train)

    feature_cls_prob = clf_bayes.feature_log_prob_
    feature_nb_weight = feature_cls_prob[1] / feature_cls_prob[0]

    doc2vec.apply_nb_weights(feature_nb_weight)
    #------------------------------Fitting naive bayes finished------------------------------

    X_train = doc2vec.get_X(X_train_start, X_train_len, mode='d2vNB')
    


    logreg_model = LogisticRegression(penalty='l2', max_iter=1000, solver='liblinear')
    log_border = 3
    C_values = np.logspace(-log_border, log_border, 15)
    params = {'C': C_values}
    gs_clf = GridSearchCV(logreg_model, params, cv=10, n_jobs=4, verbose=1)
    gs_clf.fit(X_train, y_train)

    if gs_clf.best_params_['C'] in (C_values[0], C_values[-1]):
        print("C is on border!")
        log_border += 2
        C_values = np.logspace(-log_border, log_border, 15)
        params = {'C': C_values}
        gs_clf = GridSearchCV(logreg_model, params, cv=10, n_jobs=4, verbose=1)
        gs_clf.fit(X_train, y_train)

    best_model = gs_clf.best_estimator_
    return best_model, doc2vec, datasets_info
    # ############################ REPLACE THIS WITH YOUR CODE #############################


def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    # ############################ PUT YOUR CODE HERE #######################################
    datasets_info = {}
    cur_sent = 0
    for text in texts_list:
        datasets_info[text[0]] = (cur_sent, len(text))
        cur_sent += len(text)

    chained_texts = list(chain(*texts_list))

    print("START TEXTS PREPROCESSING")
    start = time.time()
    preprocessed_texts = preprocess_texts(chained_texts)
    ngram_texts = make_ngram_dataset(preprocessed_texts)

    print(f"Finish preprocessing in {time.time() - start} seconds.")
    print("len(ngram_texts)", len(ngram_texts))


    w2ind_full, vocab_probs = make_vocab(ngram_texts, min_tf=3, max_df=0.6, min_df=3, max_tokens=1000000)
    print("len(w2ind_full)", len(w2ind_full))
    inds_texts = make_inds_ngram_dataset(ngram_texts, w2ind_full)

    doc2vec_model = Doc2Vec(len(w2ind_full), len(ngram_texts))

    #------------------------------Fitting bag of words started------------------------------
    print()
    print("Started TFIDF")
    vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.6, min_df=3, vocabulary=w2ind_full, stop_words=my_stop_words)
    X_full_bow = vectorizer.fit_transform(chained_texts)
    print("bow.shape: ", X_full_bow.shape)
#     X_train_bow = X_full_bow[:15000]
#     X_dev_bow = X_full_bow[15000: 15000 + 10000]
    
    doc2vec_model.set_bow_vectors(X_full_bow)
    print("Finish TFIDF")
    #------------------------------Fitting bag of words finished------------------------------

    d2v_epochs = 10
    base_d2v_lr = 0.07
    d2v_batch_size = 100
    d2v_nb = 5

    total_epoch_iterations = ((d2v_nb + 1) * len(inds_texts[0])) // d2v_batch_size

    loss_stat_border = 500000

    cur_iter = 0
    total_iter = d2v_epochs * total_epoch_iterations
    for ep in range(d2v_epochs):
        batch_gen = batch_generator(inds_texts[0], inds_texts[1], vocab_probs, nb=d2v_nb, batch_size=d2v_batch_size)
        avg_loss = 0.0
        for ind, (word_inds, doc_inds, labels) in enumerate(tqdm(batch_gen, total = total_epoch_iterations)):
            d2v_lr = base_d2v_lr * (1 - cur_iter * 1.0 / total_iter)
            batch_loss = doc2vec_model.train(word_inds, doc_inds, labels, d2v_lr)
            cur_iter += 1
            avg_loss += batch_loss
            if ind % loss_stat_border == 0 and ind != 0:
                tqdm.write(f"avg_loss: {avg_loss / loss_stat_border}")
                avg_loss = 0.0

    return doc2vec_model, datasets_info


def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    # ############################ REPLACE THIS WITH YOUR CODE #############################
    best_model, doc2vec, datasets_info = params

    X_test_start, X_test_len = datasets_info[texts[0]]
    X_test = doc2vec.get_X(X_test_start, X_test_len, mode='d2vNB')

    preds_int = best_model.predict(X_test)
    preds = ['pos' if pr == 1 else 'neg' for pr in preds_int]

    return preds
    # ############################ REPLACE THIS WITH YOUR CODE #############################
