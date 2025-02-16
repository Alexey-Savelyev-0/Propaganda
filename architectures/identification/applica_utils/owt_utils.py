import re

import datasets
import numpy as np
from datasets import load_dataset
#from input_processing import get_sentence_tokens_labels, get_owt_data, get_data
NUM_ARTICLES = 100
#NUM_ARTICLES = min(NUM_ARTICLES, 10)

def get_owt_articles(dataset="Skylion007/openwebtext"):
    dataset = load_dataset(dataset,trust_remote_code=True)
    # return first 100 examples
    return dataset['train'][:NUM_ARTICLES]




"""
articles = get_owt_articles()['text']

indices = np.arange(NUM_ARTICLES)
print(articles[1])
spans = [[] for _ in range(NUM_ARTICLES)]
get_data(articles,spans, indices)
"""
