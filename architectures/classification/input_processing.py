# IMPLEMENTED BY NEWSSWEEPER, SLIGHTLY ADAPTED FOR THIS PROJECT

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn import metrics
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from . import config
#import config
import torch

def get_examples(articles, spans, techniques):
  # only adds sentences with a technique - not necessarily the best approach.
  assert len(articles) == len(spans) and len(spans) == len(techniques)
  sentences = []
  labels = []
  for index, article in enumerate(articles):
    span = spans[index]
    technique = techniques[index]
    assert len(technique) == len(span)
    for i, sp in enumerate(span):
      # convert str label to int
      pt = config.tag2idx[technique[i]]
      sentence = article[sp[0]: sp[1]]
      #sentence = article[:sp[0]] + '[BOP]' + article[sp[0]:sp[1]] + '[EOP]' + article[sp[1]:]

      sentences.append(sentence)
      labels.append(pt)
  #print(sentences)
  #print(labels)
  return sentences, labels

def convert_sentence_to_input_feature(sentence, tokenizer, add_cls_sep=True, max_seq_len=150):
  tokenized_sentence = tokenizer.encode_plus(sentence,
                                             add_special_tokens=add_cls_sep,
                                             max_length=max_seq_len,
                                             truncation=True,
                                            padding='max_length',
                                             return_attention_mask=True)
  return tokenized_sentence['input_ids'], tokenized_sentence['attention_mask']

def get_data(articles, spans, techniques, shuffle):
  #print("Initial articles spans and techniques")
  #print(articles)
  #print(spans)
  #print(techniques)
  sentences, labels = get_examples(articles, spans, techniques)
  attention_masks = []
  inputs = []
  lengths = []
  for i, sentence in enumerate(sentences):
    lengths.append(len(sentence))
    """perhaps replace convert_sentence_to_inpt feature with our function?"""
    input_ids, mask = convert_sentence_to_input_feature(sentence, config.tokenizer)
    inputs.append(input_ids)
    attention_masks.append(mask)
  
  inputs = torch.tensor(inputs)
  labels = torch.tensor(labels)
  masks = torch.tensor(attention_masks)
  lengths = torch.tensor(lengths).float()
  print(inputs.shape)
  print(labels.shape)
  print(masks.shape)
  print(lengths.shape)
  tensor_data = TensorDataset(inputs, labels, masks, lengths)
  dataloader = DataLoader(tensor_data, batch_size=config.BATCH_SIZE, shuffle=shuffle)
  return dataloader

