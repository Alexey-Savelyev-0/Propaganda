import os
import torch
import numpy as np
import argparse
import wget
import gdown

import identification
import identification.hitachi_utils as hitachi_utils
from transformers import BertTokenizer
from pathlib import Path
from hitachi_si import HITACHI_SI, SLC
# parser = argparse.ArgumentParser()
# parser.add_argument('--interactive', nargs='?',  type=bool, default=False, help='Set True to enter custom input') 

# args = parser.parse_args()




def get_dev_outputs(article_dir="dev-articles"):
  test_articles, test_article_ids = identification.read_articles('dev-articles')
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

  tokenizer = hitachi_utils.tokenizer
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #model_path = os.path.join(hitachi_utils.model_dir, hitachi_utils.CURRENT_MODEL)
  model_path = Path(hitachi_utils.model_dir) / hitachi_utils.CURRENT_MODEL

  model = torch.load(model_path, map_location=hitachi_utils.device,weights_only=False)
  test_articles = ["A propaganda jihadi test to be done!", "Just a random piece of text which should be a normal text"]
  print("Starting prediction")
  
  test_spans = [[]] * len(test_articles)

  test_dataloader, test_sentences, test_bert_examples = identification.input_processing.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))

  sps = identification.pred_utils.get_score(model, 
                            dataloader=test_dataloader,
                            sentences=test_sentences,
                            bert_examples=test_bert_examples,
                            mode="test")

  for i in range(len(test_articles)):
    print(test_articles[i])
    print('Detected span: ')
    identification.utils.print_spans(test_articles[i], sps[i])
    print('--' * 50)
