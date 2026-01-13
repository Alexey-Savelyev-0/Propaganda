import os
import sys
import torch
import csv
import numpy as np
import random
import time
import datetime
import pprint
import applicaAI_tc
import classification
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split, KFold

from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    #AdamW,
    get_linear_schedule_with_warmup
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_dataloader, eval_dataloader,eval_ids,criterion, epochs=10):
  optimizer = torch.optim.AdamW(model.parameters(),lr = 1.9e-5,eps = 1e-8) # ler = 5e-5
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0, # Default value in run_glue.py
                                              num_training_steps = total_steps)
  splits = 5
  kfold = KFold(n_splits=splits, shuffle=True)
  loss_values = []
  old_acc = 0
  if classification.REWEIGHING:
    model_name = 'classification_model_reweighed_' + str(datetime.datetime.now()) + '.pt'
    print("REWEIGHING")
    print(model_name)
  else:
    model_name = 'classification_model_' + str(datetime.datetime.now()) + '.pt'
    print(model_name)
  old_step = 0
  for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
      if step % 100 == 0 and not step == 0:
        elapsed = classification.format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        print(f'step {old_step+step}')
      b_input_ids = batch[0].to(device)
      b_labels = batch[1].to(device)
      b_input_mask = batch[2].to(device)
      b_lengths = batch[3].to(device)
      model.zero_grad()        
      outputs = model(b_input_ids, 
                      token_type_ids=None, 
                      attention_mask=b_input_mask,
                    # lengths=b_lengths,
                      labels=b_labels)
      if classification.REWEIGHING == True:
        logits = outputs.logits
        loss = criterion(logits, b_labels)
      else:
        loss = outputs.loss
      total_loss += loss.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()
    old_step+= step
     
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(classification.utils.format_time(time.time() - t0)))
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in eval_dataloader:
    #for batch in train_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_labels, b_input_mask, b_lengths = batch
      with torch.no_grad():        
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
                        # lengths=b_lengths)
      
      #logits = outputs[0]
      logits = outputs.logits
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      tmp_eval_accuracy = classification.flat_accuracy(logits, label_ids)
      
      eval_accuracy += tmp_eval_accuracy
      nb_eval_steps += 1
    classification.pred_utils.get_dev_predictions(model,eval_dataloader,eval_ids)
    new_acc = eval_accuracy/nb_eval_steps
    
    print("  Accuracy: {0:.2f}".format(new_acc))
    print("  Validation took: {:}".format(classification.format_time(time.time() - t0)))
    if new_acc>old_acc:
      torch.save(model, os.path.join(classification.model_dir, model_name))
      old_acc = new_acc

  print("")
  print("Training complete!")







articles, article_ids = classification.read_articles("train-articles")
spans, techniques = classification.read_spans()
pprint.pprint(classification.tag2idx)

NUM_ARTICLES = len(articles)
NUM_ARTICLES = classification.NUM_ARTICLES
articles = articles[0:NUM_ARTICLES]
article_ids = article_ids[0:NUM_ARTICLES]
spans = spans[0:NUM_ARTICLES]
techniques = techniques[0:NUM_ARTICLES]

seed_val = 34
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

indices = np.arange(NUM_ARTICLES)

train_articles, eval_articles, train_ids, eval_ids, train_spans, eval_spans, train_techniques, eval_techniques, train_indices, eval_indices = train_test_split(articles,article_ids, spans, techniques, indices, test_size=0.1, shuffle=False)

frequencies = [0 for i in range(0,14)]
for i in train_techniques:
  for j in i:
    pt = classification.tag2idx[j]
    frequencies[pt]+=1
frequencies = np.array(frequencies)
class_weights = max(frequencies) / frequencies
class_weights = class_weights / class_weights.sum()  
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(frequencies)
print(class_weights_tensor)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

# Normalize weights
train_dataloader = classification.get_data(train_articles, train_spans, train_techniques,shuffle=True)
eval_dataloader = classification.get_data(eval_articles, eval_spans, eval_techniques, shuffle= False)

tokenizer = classification.tokenizer

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels = len(classification.tag2idx),
    output_attentions = False,
    output_hidden_states = False,
    hidden_dropout_prob=0.1,  # default is 0.1
    attention_probs_dropout_prob=0.1
)



#model = applicaAI_tc.EnsembleModel(15)
if torch.cuda.is_available():
  print('Using GPU')
  model.cuda()
print(f"train ids are{train_ids}")
train(model, train_dataloader, eval_dataloader,eval_ids,criterion, epochs=10)

