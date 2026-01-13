""" predict_SI.py - runs the custom scorer provided by Da San Martino et al. to evaluate all of the models in 
si_models folder. Orginially sourced from NewsSweeper, but adapted to to run for a wider range of models.
"""



import os
import numpy as np
import argparse
import torch
import identification

from transformers import BertTokenizer
from pathlib import Path
from hitachi_si import HITACHI_SI, SLC
import classification 
from LSTM import LSTMClassifier
from transformers import RobertaForTokenClassification

def get_dev_outputs(article_dir="dev-articles"):
  test_articles, test_article_ids = identification.read_articles('train-articles')
  test_articles, test_article_ids = test_articles[300:], test_article_ids[300:]
  test_spans = [[]] * len(test_articles)
  test_dataloader, test_sentences, test_bert_examples = identification.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))
  sps = identification.get_score(model,
      dataloader=test_dataloader,
      sentences=test_sentences,
      bert_examples=test_bert_examples,
      mode="test")
  with open('dev_predictions.txt', 'w') as fp:
    for index in range(len(test_articles)):
      for ii in sps[index]:
        fp.write(test_article_ids[index] + "\t" + str(ii[0]) + "\t" + str(ii[1]) + "\n")

def get_predictions(text):
  test_articles = [text]
  test_spans = [[]] * len(test_articles)
  test_dataloader, test_sentences, test_bert_examples = identification.input_processing.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))
  sps = identification.pred_utils.get_score(model, 
                            dataloader=test_dataloader,
                            sentences=test_sentences,
                            bert_examples=test_bert_examples,
                            mode="test")

  spans = identification.utils.return_spans(test_articles[0], sps[0])
  return spans

def get_predictions_indices(text):
  test_articles = [text]
  test_spans = [[]] * len(test_articles)
  test_dataloader, test_sentences, test_bert_examples = identification.input_processing.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))
  sps = identification.pred_utils.get_score(model, 
                            dataloader=test_dataloader,
                            sentences=test_sentences,
                            bert_examples=test_bert_examples,
                            mode="test")

  return sps[0]



if __name__ == "__main__":

  tokenizer = identification.tokenizer
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # get all the model names and evaluate them seqeuntailly

  articles, article_ids = identification.read_articles("train-articles")
  """ note that the spans outlined in the classification section are 
    similar but different-> additional ones are added in the classification section.
    Therefore only use spans/techniques if span exists in identification section"""
    
  spans = identification.read_spans()
  tc_spans, techniques = classification.read_spans()
  indices = np.arange(len(spans))
  indices = indices[300:]
  tc_spans_techniques = {}
  for i in range(len(tc_spans)):
        for j in range(len(tc_spans[i])):
            tc_spans_techniques[(i,tc_spans[i][j])] = techniques[i][j]
        
  techniques = identification.get_si_techniques(spans,tc_spans_techniques)

    
  #NUM_ARTICLES = identification.NUM_ARTICLES
  
  indices = np.arange(370)
  eval_indices = indices[300:]
    #silver_articles = identification.get_owt_articles()['text']

  test_dataloader, test_sentences, test_bert_examples = identification.get_data_bert(articles, spans, eval_indices,techniques=techniques)

    #dataloader = identification.get_data(articles, spans, techniques, train_indices, PLM = hitachi_si.PLM, s = hitachi_si.s, c = hitachi_si.c)




  folder = Path(identification.model_dir)

# run test an all files in directory
  model_names = [p.name for p in folder.iterdir() if p.is_file()]
  torch.serialization.add_safe_globals([HITACHI_SI])
  for name in model_names:
    
    if name[:7]== "roberta":
       print(f"Loading RoBERTa:{name}")
       
       model_type = 'bert'
    elif name[:7]=="hitachi":
       print(f"Loading main model: {name}")
       
       model_type = 'n/a'
    else:
       print(f'Unknown {name[:8]}')

    model_path = Path(identification.model_dir) / name
    model = torch.load(model_path, map_location=identification.device,weights_only=False)
    model = model.to(device)

    


    sps = identification.pred_utils.get_score(model, 
                              dataloader=test_dataloader,
                              sentences=test_sentences,
                              bert_examples=test_bert_examples,indices=eval_indices,
                              mode="eval",article_ids=article_ids,model_type=model_type)
