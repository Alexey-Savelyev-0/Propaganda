import time
import datetime
from tqdm import tqdm, trange

import torch
import numpy as np
import os
from transformers import BertTokenizer, BertForTokenClassification
from transformers import get_linear_schedule_with_warmup, AdamW
#from torch import AdamW as AdamWnew
import identification
import identification.applica_utils as applica_utils


def train(model, train_dataloader, eval_dataloader, epochs=5, save_model=False):
  max_grad_norm = 1.0

  for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
      # add batch to gpu
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_labels, b_input_mask, b_ids = batch

      output = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask, labels=b_labels)
      output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
      loss= output.get("loss")
      loss.backward()
      tr_loss += loss.item()
      nb_tr_examples += b_input_ids.size(0)
      nb_tr_steps += 1
      torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
      optimizer.step()
      # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
      model.zero_grad()
    print("Train loss: {}".format(tr_loss/nb_tr_steps))


    identification.get_score(model,
        train_dataloader,
        train_sentences,
        train_bert_examples,
        mode="train",
        article_ids=article_ids,
        indices=train_indices)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in eval_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_labels, b_input_mask, b_ids = batch
      with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        tmp_eval_loss = outputs.get("loss")  # Use `.loss` if using a HuggingFace model
        logits = outputs.get("logits")  # Use `.logits` for logits   
      eval_loss += tmp_eval_loss.mean().item()
      
      
      nb_eval_examples += b_input_ids.size(0)
      nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))

    identification.get_score(model,
        eval_dataloader,
        eval_sentences,
        eval_bert_examples,
        mode="eval",
        article_ids=article_ids,
        indices=eval_indices)
    if save_model:
      model_name = 'model_' + str(datetime.datetime.now()) + '.pt'
      torch.save(model, os.path.join(applica_utils.model_dir, model_name))
      print("Model saved:", model_name)
    print()
    time.sleep(1)


articles, article_ids = identification.read_articles('train-articles')
spans = identification.read_spans()
NUM_ARTICLES = len(articles)

NUM_ARTICLES = min(NUM_ARTICLES, 5)
articles = articles[0:NUM_ARTICLES]
spans = spans[0:NUM_ARTICLES]

np.random.seed(245)
indices = np.arange(NUM_ARTICLES)
print(indices)
np.random.shuffle(indices)
train_indices = indices[:int(0.9 * NUM_ARTICLES)]
print(train_indices)

eval_indices = indices[int(0.9 * NUM_ARTICLES):]
print(eval_indices)
tokenizer = applica_utils.tokenizer
TAGGING_SCHEME = applica_utils.TAGGING_SCHEME
BATCH_SIZE = applica_utils.BATCH_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader, train_sentences, train_bert_examples = identification.get_data(articles, spans, train_indices)
eval_dataloader, eval_sentences, eval_bert_examples = identification.get_data(articles, spans, eval_indices)


num_labels = 2 + int(TAGGING_SCHEME =="BIO") + 2 * int(TAGGING_SCHEME == "BIOE")
if applica_utils.LANGUAGE_MODEL == "RoBERTa":
  from transformers import RobertaForTokenClassification
  model = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=num_labels)
elif applica_utils.LANGUAGE_MODEL == "RoBERTa-CRF":
  from transformers import RobertaForTokenClassification
  # for now use roberta base
  model_base = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=num_labels)
  model = applica_utils.RobertaCRF(model_base, num_labels)
else:
  from transformers import BertForTokenClassification
  model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

if torch.cuda.is_available():
  print('Using cuda')
  model.cuda()

if TAGGING_SCHEME == "BIOE":
  WEIGHTS = torch.tensor([1.0, 5.0, 10.0, 5.0])
  if torch.cuda.is_available():
    WEIGHTS = WEIGHTS.cuda()
else:
  WEIGHTS = torch.tensor([1.0, 100.0])
  if torch.cuda.is_available():
    WEIGHTS = WEIGHTS.cuda()

epochs = 4
total_steps = total_steps = len(train_dataloader) * epochs

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_steps = total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

train(model, train_dataloader, eval_dataloader, epochs=epochs, save_model=(NUM_ARTICLES >= 150))
