import numpy as np
import os
import csv
import datasets
from datasets import load_dataset
from .hitachi_utils import config
data_dir = config.data_dir


def get_owt_articles(dataset="Skylion007/openwebtext",num_articles=100):
    dataset = load_dataset(dataset,trust_remote_code=True)
    # return first 100 examples
    return dataset['train'][:num_articles]

def read_articles(article_dir):
  articles = []
  train_dir = os.path.join(data_dir, article_dir)
  print(os.getcwd())
  for filename in sorted(os.listdir(train_dir)):
    myfile = open(os.path.join(train_dir, filename),encoding="utf8")
    article = myfile.read()
    articles.append(article)
    myfile.close()
  article_ids = []
  for filename in sorted(os.listdir(train_dir)):
    article_ids.append(filename[7:-4])
  return articles, article_ids

def read_spans():
  spans = []
  label_dir = os.path.join(data_dir, "train-labels-task1-span-identification")
  for filename in sorted(os.listdir(label_dir)):
    myfile = open(os.path.join(label_dir, filename))
    tsvreader = csv.reader(myfile, delimiter="\t")
    span = []
    for row in tsvreader:
      span.append((int(row[1]), int(row[2])))
    myfile.close()
    spans.append(span)
  return spans

def print_spans(article, span):
  for sp in span:
    print (article[sp[0]: sp[1]])
  print()

def return_spans(article, span):
  spans = []
  for sp in span:
    spans.append(article[sp[0] : sp[1]])
  return spans

def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=2).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)
