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
import identification.hitachi_utils as hitachi_utils
import classification.hitachi_utils as classification
import identification.applica_utils as applica_utils


def train(model, train_dataloader, eval_dataloader, epochs=5, save_model=True):
  max_grad_norm = 1.0

  for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
      # add batch to gpu
      
      b_input_ids, b_labels,b_masks,b_ids, b_techniques = batch
      for i in range(len(b_labels)):
        assert(len([j for j in b_labels[i] if j == 1 or j == 2 or j== 0])) == len([j for j in b_masks[i] if j != 0])
      b_input_ids = b_input_ids.to(device)
      b_labels = b_labels.to(device)
      b_masks = b_masks.to(device)
      b_ids = b_ids.to(device)
      b_techniques = b_techniques.to(device)
      output = model(b_input_ids, token_type_ids=None,attention_mask=b_masks, labels=b_labels)
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
      b_input_ids, b_labels,b_masks,b_ids, b_techniques = batch
      with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_masks, labels=b_labels)
        tmp_eval_loss = outputs.get("loss")  # Use `.loss` if using a HuggingFace model
        logits = outputs.get("logits")  # Use `.logits` for logits   
      eval_loss += tmp_eval_loss.mean().item()
      
      
      nb_eval_examples += b_input_ids.size(0)
      nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    """
    identification.get_score(model,
        eval_dataloader,
        eval_sentences,
        eval_bert_examples,
        mode="eval",
        article_ids=article_ids,
        indices=eval_indices)
    """
  if save_model:
      model_name = 'roberta_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.pt'
      torch.save(model, os.path.join(applica_utils.model_dir, model_name))
      print("Model saved:", model_name)


articles, article_ids = identification.read_articles('train-articles')
spans = identification.read_spans()
tc_spans, techniques = classification.read_spans()
tc_spans_techniques = {}
for i in range(len(tc_spans)):
    for j in range(len(tc_spans[i])):
      tc_spans_techniques[(i,tc_spans[i][j])] = techniques[i][j]
        
techniques = identification.get_si_techniques(spans,tc_spans_techniques)
NUM_ARTICLES = len(articles)

NUM_ARTICLES = min(NUM_ARTICLES, hitachi_utils.NUM_ARTICLES)
articles = articles[0:NUM_ARTICLES]
spans = spans[0:NUM_ARTICLES]
techniques = techniques[0:NUM_ARTICLES]
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

train_dataloader, train_sentences, train_bert_examples = identification.get_data(articles, spans, train_indices,techniques)
eval_dataloader, eval_sentences, eval_bert_examples = identification.get_data(articles, spans, eval_indices,techniques)


num_labels = 2 + int(TAGGING_SCHEME =="BIO") + 2 * int(TAGGING_SCHEME == "BIOE")
if hitachi_utils.LANGUAGE_MODEL == "RoBERTa":
  from transformers import RobertaForTokenClassification
  model = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=num_labels)
elif hitachi_utils.LANGUAGE_MODEL == "RoBERTa-CRF":
  from transformers import RobertaForTokenClassification
  # for now use roberta base
  model_base = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=num_labels)
  model = hitachi_utils.RobertaCRF(model_base, num_labels)
else:
  from transformers import BertForTokenClassification
  model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

if torch.cuda.is_available():
  print('Using cuda')
  model.cuda()


#else:
  #WEIGHTS = torch.tensor([1.0, 100.0])
  #if torch.cuda.is_available():
    #WEIGHTS = WEIGHTS.cuda()

epochs = hitachi_utils.EPOCHS
total_steps = total_steps = len(train_dataloader) * epochs

optimizer = torch.optim.AdamW(model.parameters(), lr=hitachi_utils.LEARNING_RATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_steps = total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

train(model, train_dataloader, eval_dataloader, epochs=epochs, save_model=True)
