# IMPLEMENTED BY NEWSSWEEPER, SLIGHTLY ADAPTED FOR THIS PROJECT


import os
import torch
import numpy as np
from shutil import copyfile
from . import config
from pathlib import Path
def merge_spans(current_spans):
  if not current_spans:
    return [] 
  merged_spans = []
  li = current_spans[0][0]
  ri = current_spans[0][1]
  threshold = 2
  for i in range(len(current_spans) - 1):
    span = current_spans[i+1]
    if span[0] - ri < threshold:
      ri = span[1]
      continue
    else:
      merged_spans.append((li, ri))
      li = span[0]
      ri = span[1]
  merged_spans.append((li, ri))
  return merged_spans


def get_model_predictions(model, dataloader,model_type = 'n/a'):
  model.eval()
  predictions , true_labels, sentence_ids = [], [], []
  nb_eval_steps = 0
  for batch in dataloader:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make t a tensor first

    #batch = tuple(t.to(device) for t in batch)
    b_input_ids,b_labels,b_context,b_masks,b_ids, b_techniques, b_mapping = batch  
    b_input_ids = b_input_ids.to(device)
    b_masks = b_masks.to(device)
    b_labels = b_labels.to(device)     
    b_techniques = b_techniques.to(device)   
    b_context = b_context.to(device)
    b_mapping = b_mapping.to(device)
    with torch.no_grad():
      if model_type == 'n/a':
        logits = model(b_input_ids, 
                          token_type_ids = None,
                          labels_TC = None,
                          token_enrichment = b_context,
                          token_mapping = b_mapping,
                          attention_mask = b_masks)
        predictions.extend([list(p) for p in logits])
        #print(predictions)
      else:
        logits = model(b_input_ids, token_type_ids=None,attention_mask=b_masks)
        #print(logits.logits)
        prediction = torch.argmax(logits.logits, dim=-1)
        #print(prediction)
        predictions.extend([list(p) for p in prediction])
      
      
    label_ids = b_labels.to('cpu').numpy()
    s_ids = b_ids.to('cpu').numpy()
    
    true_labels.extend(label_ids)
    sentence_ids.extend(s_ids)
    nb_eval_steps += 1
  return predictions, true_labels, sentence_ids


from typing import List, Union
def parse_logits(logits: Union[List[List[int]],
                             dict,
                             torch.Tensor,
                             np.ndarray]) -> List[List[int]]:
    """
    Normalize various logits outputs into List[List[int]].
    Safely handles 0-d tensors/arrays by expanding them.
    """
    if isinstance(logits, list):
        return logits

    if isinstance(logits, dict) and "logits" in logits:
        logits = logits["logits"]

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    logits = np.asarray(logits)

    if logits.ndim == 0:
        logits = logits.reshape(1, 1)

    predictions: List[List[int]] = []
    for row in logits:
        predictions.append(row.tolist())

    return predictions


def get_score(model, dataloader, sentences, bert_examples, mode=None, article_ids=None, indices=None,model_type = 'bert'):
  predicted_spans = [[] for i in range(4000)] 
  
  def get_span_prediction(prediction_labels, sentence_index, sentences, bert_examples):
    index = sentence_index 
    bert_example = bert_examples[index]
    mask = bert_example.input_mask

    pred_labels_masked = prediction_labels 
    pred_labels = []
    """
    print(sentences[index].tokens)
    print(bert_example.tokens)
    print(prediction_labels)
    #print(len(prediction_labels))
    print(mask)
    """
    for i, m in enumerate(mask):
      if m > 0:
        pred_labels.append(pred_labels_masked[i])
    if bert_example.add_cls_sep:
      pred_labels.pop() # remove ['SEP'] label
      pred_labels.pop(0) # remove ['CLS'] label

    sentence = sentences[index]
    sent_len = len(sentence.tokens)
    final_pred_labels = [0] * (sent_len-2)
    cur_map = bert_example.tok_to_orig_index[1:-1]
    for i, label in enumerate(pred_labels):
      try:
        if cur_map[i]< len(final_pred_labels):
          final_pred_labels[cur_map[i]] |= label
      except:
        print("EXCEPTION")
        print(i)
        print(cur_map)
        print(cur_map[i])
        print(final_pred_labels)
        print(len(final_pred_labels))
        raise Exception
    
    word_start_index_map = sentence.word_to_start_char_offset
    word_end_index_map = sentence.word_to_end_char_offset
    article_index = sentence.article_index
    for i, label in enumerate(final_pred_labels):
      if label:
        predicted_spans[article_index].append((word_start_index_map[i], word_end_index_map[i]))

  predictions, _, sentence_ids = get_model_predictions(model, dataloader,model_type=model_type)
  pred_sentences, pred_bert_examples = sentences, bert_examples
  merged_predicted_spans = []
  for ii, _ in enumerate(predictions):
    get_span_prediction(predictions[ii], sentence_ids[ii], pred_sentences, pred_bert_examples)
  for span in predicted_spans:
    merged_predicted_spans.append(merge_spans(span))

  if mode == "test":
    return merged_predicted_spans 
  #print("MERGED SPANS")
  #print(merged_predicted_spans)
  if not os.path.isdir("predictions"):
    os.mkdir("predictions")
  src = Path(config.home_dir) / "architectures"/"tools" / "task-SI_scorer.py"
  dst = Path("predictions") / "task-SI_scorer.py"

  copyfile(src, dst)
  with open("predictions/predictions.tsv", 'w') as fp:
    for index in indices:
      filename = "article" + article_ids[index] + ".task1-SI.labels"
      #print(f"checking file{filename}")
      copyfile(os.path.join(config.data_dir, "train-labels-task1-span-identification/" + filename), "predictions/" + filename)
      for ii in merged_predicted_spans[index]:
        point1 = int(ii[0])
        point2= int(ii[1])
        if point1>point2:
          point1,point2 = point2,point1
        fp.write(article_ids[index] + "\t" + str(point1) + "\t" + str(point2) + "\n")

  os.system("python3 predictions/task-SI_scorer.py -s predictions/predictions.tsv -r predictions/ -m")

  for index in indices:
    filename = "article" + article_ids[index] + ".task1-SI.labels"
    os.remove("predictions/" + filename)

