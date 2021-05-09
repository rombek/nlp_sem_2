import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

from collections import defaultdict
from functools import partial
from itertools import chain
import re
import string
import time
from tqdm.notebook import trange, tqdm

from filimdb_evaluation.score import load_dataset_fast



def main():
	texts = ['hello it is me mario', 'does it make me strong']
	w2ind = dict()
	i = 0
	for text in texts:
		for w in text.split():
			if w not in w2ind:
				w2ind[w] = i
				i += 1

	print(w2ind)
	vectorizer_base = TfidfVectorizer()
	vec_texts1 = vectorizer_base.fit_transform(texts)

	vectorizer_my = TfidfVectorizer(vocabulary=w2ind)
	vec_texts2 = vectorizer_my.fit_transform(texts)

	print("FIRST: ")
	print(vec_texts1)

	print("MY: ")
	print(vec_texts2)

if __name__ == '__main__':
	main()