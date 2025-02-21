import torch

from . import config
#from . import config

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class example_sentence:
  def __init__(self):
    self.tokens = []
    self.labels = []
    self.article_index = -1 # index of the article to which the sentence is associated
    self.index = -1 # index of the sentence in that article 
    self.word_to_start_char_offset = []
    self.word_to_end_char_offset = []
  
  def __str__(self):
    print("tokens -", self.tokens)
    print("labels -", self.labels)
    print("article_index -", self.article_index)
    print("index -", self.index)
    print("start_offset -", self.word_to_start_char_offset)
    print("end_offset -", self.word_to_end_char_offset)   
    return "" 

def is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    return True
  return False

def get_sentence_tokens_labels(article, span=None, article_index=None):
  doc_tokens = [] # builds up to a word
  char_to_word_offset = []
  all_sentence_tokens = []
  word_to_start_char_offset = {}
  word_to_end_char_offset = {}
  new_word_to_start_char_offset = {}
  new_word_to_end_char_offset = {}
  prev_is_whitespace = True
  prev_is_newline = True
  current_word_position = None
  max_seq_length = 100
  # can be updated with spacy
  import spacy
  nlp = spacy.load("en_core_web_sm")
  """
  # TEST THIS DOESNT BREAK FUNCTION
  sentences = article.split("\n")
  for x in range(len(sentences)):
    doc = nlp(sentences[x])
    doc_tokens = [token.text for token in doc]
    all_sentence_tokens.append(doc_tokens)
    for token in doc:
      new_word_to_start_char_offset[(x, token.i)] = token.idx
      new_word_to_end_char_offset[(x, token.i)] = token.idx + len(token.text)
  word_to_start_char_offset = {}
  """
  
  for index, c in enumerate(article):
    if len(doc_tokens) >= max_seq_length:
      #current_word_position = (len(all_sentence_tokens), len(doc_tokens) - 1)
      prev_is_whitespace = True
      word_to_end_char_offset[current_word_position] = index
      all_sentence_tokens.append(doc_tokens)
      doc_tokens = []

      
      #if current_word_position is not None:
        #current_word_position = (len(all_sentence_tokens), len(doc_tokens) - 1)
        #word_to_start_char_offset[current_word_position] = index
        #current_word_position = None
      
      #word_to_start_char_offset[current_word_position] = index
      
    if c == "\n":
      prev_is_newline = True
      # check for empty lists
      if doc_tokens:
        all_sentence_tokens.append(doc_tokens)
        doc_tokens = []
    if is_whitespace(c):
      prev_is_whitespace = True
      if current_word_position is not None:
        word_to_end_char_offset[current_word_position] = index
        current_word_position = None
    else:
      if prev_is_whitespace: # if whitespace, start a new word
        doc_tokens.append(c)
        current_word_position = (len(all_sentence_tokens), len(doc_tokens) - 1)
        word_to_start_char_offset[current_word_position] = index # start offset of word
      else: # add to the current word
        if doc_tokens:
          doc_tokens[-1] += c
        else:
          doc_tokens.append(c)
      prev_is_whitespace = False
    char_to_word_offset.append((len(all_sentence_tokens), len(doc_tokens) - 1)) # maps current sentence to length of doc_tokens. 
    # in other words, keeps log of how long each word is for a sentence 
  if doc_tokens:
    all_sentence_tokens.append(doc_tokens)
  

  if current_word_position is not None:
    word_to_end_char_offset[current_word_position] = index
    current_word_position = None
  

  if span is None:
    return all_sentence_tokens, (word_to_start_char_offset, word_to_end_char_offset)

  current_propaganda_labels = []
  for doc_tokens in all_sentence_tokens:
    current_propaganda_labels.append([0] * len(doc_tokens))

  start_positions = []
  end_positions = []

  for sp in span:
    
    if (char_to_word_offset[sp[0]][0] != char_to_word_offset[sp[1]-1][0]):
      # if the sentence of the character at the beggining of the span isn't the same as sentence of the character at the end of the span.
      l1 = char_to_word_offset[sp[0]][0] # sentence 1
      l2 = char_to_word_offset[sp[1] - 1][0] # sentence 2
      start_positions.append(char_to_word_offset[sp[0]]) 
      # add (sentence, index of curr word) of starting word for span.
      end_positions.append((l1, len(all_sentence_tokens[l1])-1))
      # add (sentence 1, lenth of sentence 1) i.e say that the span goes to the end of the sentence.
      l1 += 1
      while(l1 < l2):
        start_positions.append((l1, 0)) # add another span starting at next sentence from sentence 1
        end_positions.append((l1, len(all_sentence_tokens[l1])-1))
        l1 += 1
        # keep filling sentences with spans until sentence 2 reached.
      start_positions.append((l2, 0)) 
      end_positions.append(char_to_word_offset[sp[1]-1])  
      continue
    # add whichever character is at the outlined char
    start_positions.append(char_to_word_offset[sp[0]])
    end_positions.append(char_to_word_offset[sp[1]-1])

  for i, _ in enumerate(start_positions):
    assert start_positions[i][0] == end_positions[i][0]
    if config.TAGGING_SCHEME == "BIO":
      current_propaganda_labels[start_positions[i][0]][start_positions[i][1]] = 2 # Begin label
      if start_positions[i][1] < end_positions[i][1]:
        current_propaganda_labels[start_positions[i][0]][start_positions[i][1] + 1 : end_positions[i][1] + 1] = [1] * (end_positions[i][1] - start_positions[i][1])
      
  num_sentences = len(all_sentence_tokens)

  start_offset_list = get_list_from_dict(num_sentences, word_to_start_char_offset)
  end_offset_list = get_list_from_dict(num_sentences, word_to_end_char_offset)
  sentences = []
  for i in range(num_sentences):
    sentence = example_sentence()
    sentence.tokens = all_sentence_tokens[i]
    sentence.labels = current_propaganda_labels[i]
    sentence.article_index =  article_index
    sentence.index = i
    
    sentence.word_to_start_char_offset = start_offset_list[i]
    sentence.word_to_end_char_offset = end_offset_list[i]
    
    num_words = len(sentence.tokens)
    assert len(sentence.labels) == num_words
    assert len(sentence.word_to_start_char_offset) == num_words
    assert len(sentence.word_to_end_char_offset) == num_words
    sentences.append(sentence)

  return all_sentence_tokens, current_propaganda_labels, (word_to_start_char_offset, word_to_end_char_offset), sentences

def get_list_from_dict(num_sentences, word_offsets):
  li = []
  for _ in range(num_sentences):
    li.append([])
  for key in word_offsets:
    si = key[0]
    li[si].append(word_offsets[key])

  return li

class BertExample:
  def __init__(self):
    self.add_cls_sep = True
    self.sentence_id = -1
    self.orig_to_tok_index = []
    self.tok_to_orig_index = []
    self.labels = None
    self.tokens_ids = []
    self.input_mask = []
  def __str__(self):
    print("sentence_id", self.sentence_id)
    return ""

def convert_sentence_to_input_feature(sentence, sentence_id, tokenizer, add_cls_sep=True, max_seq_len=256):
  bert_example = BertExample()
  bert_example.sentence_id = sentence_id
  bert_example.add_cls_sep = add_cls_sep

  sentence_tokens = sentence.tokens
  sentence_labels = sentence.labels 

  # index of which subtokens are part of which word
  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = [] 
  for (i, token) in enumerate(sentence_tokens):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      tok_to_orig_index.append(i)
      all_doc_tokens.append(sub_token)
  bert_example.tok_to_orig_index = tok_to_orig_index
  bert_example.orig_to_tok_index = orig_to_tok_index

  bert_tokens = all_doc_tokens
  if add_cls_sep:
    bert_tokens = ["[CLS]"] + bert_tokens
    bert_tokens = bert_tokens + ["[SEP]"]
  
  tokens_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
  input_mask = [1] * len(tokens_ids)
  while len(tokens_ids) < max_seq_len:
    tokens_ids.append(0)
    input_mask.append(0)
  # tokens_ids = pad_sequences(tokens_ids, maxlen=max_seq_len, truncating="post", padding="post", dtype="int")
  bert_example.tokens_ids = tokens_ids
  bert_example.input_mask = input_mask
  # bert_example.input_mask = [float(i>0) for i in token_ids]

  if sentence_labels is None:
    return bert_example
  

  labels = [0] * len(all_doc_tokens)
  # the label of each word is replicated for each subtoken of the word
  for index, token in enumerate(all_doc_tokens):
    labels[index] = sentence_labels[tok_to_orig_index[index]]
  if add_cls_sep:
    labels = [0] + labels
    labels = labels + [0]
  # labels = pad_sequences(labels, maxlen=max_seq_len, truncating="post", padding="post", dtype="int")
  while len(labels) < max_seq_len:
    labels.append(0)
  bert_example.labels = labels

  return bert_example 

def get_dataloader(examples):
  for d in examples:
    if len(d.tokens_ids) > 256:
      print("Sentence length greater than 256")
      print(len(d.tokens_ids))
  inputs = torch.tensor([d.tokens_ids for d in examples])
  print(inputs.shape)
  labels = torch.tensor([d.labels for d in examples])
  print(labels.shape)
  masks = torch.tensor([d.input_mask for d in examples])
  sentence_ids = torch.tensor([d.sentence_id for d in examples])
  tensor_data = TensorDataset(inputs, labels, masks, sentence_ids)
  dataloader = DataLoader(tensor_data, batch_size=config.BATCH_SIZE)
  return dataloader



def get_data(articles, spans, indices):
  assert len(articles) == len(spans)    
  sentences = []
  for index in indices:
    article = articles[index]
    span = spans[index]
    _, _, _, cur_sentences = get_sentence_tokens_labels(article, span, index)
    sentences += cur_sentences
    #sentences[list[Sentence]]
  
  
  
  bert_examples = []
  # currently one of the sentences is greater than 256 tokens - if this is the case, we need to split the sentence

  for i, sentence in enumerate(sentences):
    if len(sentence.tokens) > 256:
      print("Sentence length greater than 256 aa")
      print(len(sentence.tokens))
    input_feature = convert_sentence_to_input_feature(sentence, i, config.tokenizer)
    bert_examples.append(input_feature)
  # if len(bert_examples) > 256: split
  dataloader = get_dataloader(bert_examples)
  
  return dataloader, sentences, bert_examples

def get_owt_data(articles,indices):
  sentences = []
  for index in indices:
    article = articles[index]
    cur_sentences, offsets = get_sentence_tokens_labels(article, None, index)
    sentences += cur_sentences

  bert_examples = []
  for i, sentence in enumerate(sentences):
    input_feature = convert_sentence_to_input_feature(sentence, i, config.tokenizer)
    bert_examples.append(input_feature)
  dataloader = get_dataloader(bert_examples)
  return dataloader, sentences, bert_examples

