""" Utility functions: slightly tweaked to avoid bugs/crashing but in large designed by team NewsSweeper.

"""

import torch
from . import config
import spacy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_sequence
import torch.nn.functional as F
from transformers import BertTokenizerFast

device = config.device


nlp = spacy.load("en_core_web_sm")

# extract the full list of possible POS and NER labels
pos_labels = list(nlp.get_pipe("tagger").labels)        # e.g. ["NN","VBZ",…] :contentReference[oaicite:0]{index=0}
ner_labels = list(nlp.get_pipe("ner").labels)           # e.g. ["PERSON","ORG",…] :contentReference[oaicite:1]{index=1}
# create lookup dicts
pos2id = {lbl: i for i, lbl in enumerate(pos_labels)}
ner2id = {lbl: i for i, lbl in enumerate(ner_labels)}
pos2id['CLS'] = 0
pos2id['SEP'] = 0
pos2id['None'] = 0
ner2id['SEP'] = 0
ner2id['CLS'] = 0
ner2id['None'] = 0
ner2id[""] = 0

class ExampleSentence:
  def __init__(self):
    self.tokens = [] # tokens contained in sentence
    self.length = 0  # length of the sentence
    self.labels = [] # labels for the sentence
    self.techniques = [] # technique labels for the sentence
    self.tok2idx=[]
    self.ner = []
    self.pos= []
    self.sentence = None
    self.article_index = -1 # index of the article to which the sentence is associated
    self.index = -1 # index of the sentence in that article 
    self.word_to_start_char_offset = []
    self.word_to_end_char_offset = []
  
  def __str__(self):
    print("tokens -", self.tokens)
    print("labels -", self.labels)
    print("article_index -", self.article_index)
    print("techniques - ", self.techniques)
    print("index -", self.index)
    print("start_offset -", self.word_to_start_char_offset)
    print("end_offset -", self.word_to_end_char_offset)   
    return "" 

def is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    return True
  return False


distinct_techniques = [
 'Flag-Waving',
 'Name_Calling,Labeling',
 'Causal_Oversimplification',
 'Loaded_Language',
 'Appeal_to_Authority',
 'Slogans',
 'Appeal_to_fear-prejudice',
 'Exaggeration,Minimisation',
 'Bandwagon,Reductio_ad_hitlerum',
 'Thought-terminating_Cliches',
 'Repetition',
 'Black-and-White_Fallacy',
 'Whataboutism,Straw_Men,Red_Herring',
 'Doubt'
]
distinct_techniques.insert(0, 'Non_propaganda')
tag2idx = {t: i for i, t in enumerate(distinct_techniques)}

def get_si_techniques(si_spans,tc_spans_techniques):
    """
    Get techniques for the spans described in SI training data
    i: article index
    j: span index

    returns: si_techniques, [[str]]
    """
    si_techniques=[]
    for i in range(len(si_spans)):
        si_techniques.append([0]*len(si_spans[i]))
        for j in range(len(si_spans[i])):
            if (i,si_spans[i][j]) in tc_spans_techniques:
                si_techniques[i][j] = tc_spans_techniques[(i,si_spans[i][j])]
    for i in range(len(si_spans)):
        assert len(si_techniques[i]) == len(si_spans[i])
    return si_techniques


def get_data_bert(articles, spans, indices,techniques=None,return_dataset=False,shuffle="error"):
  assert len(articles) == len(spans)    
  sentences = []
  for index in indices:
    article = articles[index]
    span = spans[index]
    article_techniques = techniques[index] if techniques != None else None
    _, _, _, cur_sentences = get_example_bert_sentences(article, span, index, article_techniques)
    sentences += cur_sentences
  
  
  
  bert_examples = []

  for i, sentence in enumerate(sentences):
    if len(sentence.tokens) > 256:
      print("Sentence length greater than 256 ")
      print(len(sentence.tokens))
    input_feature = convert_sentence_to_input_feature(sentence, i)
    
    bert_examples.append(input_feature)
  # stack if working with a list of tensors (like embeddings), use torch.tensor if working with list of lists.
  
  #for i, sentence in enumerate(sentences):
     #print(sentence.tokens)
     #print(bert_examples[i].tokens)
  inputs = torch.tensor([d.tokens_ids for d in bert_examples])
  labels = torch.tensor([d.labels for d in bert_examples])
  context = torch.stack([d.context for d in bert_examples])
  masks = torch.tensor([d.input_mask for d in bert_examples])

  sentence_ids = torch.tensor([d.sentence_id for d in bert_examples])
  bert_techniques = torch.tensor([d.techniques for d in bert_examples])
  tok2idx = torch.tensor([d.tok_to_orig_index for d in bert_examples])
  if return_dataset:
    dataloader,dataset = get_dataloader(inputs,labels,context,masks,sentence_ids,bert_techniques,tok2idx,return_dataset=return_dataset,shuffle=shuffle)
    
    return dataloader,dataset, sentences, bert_examples
  else: 
    dataloader = get_dataloader(inputs,labels,context,masks,sentence_ids,bert_techniques,tok2idx,return_dataset=return_dataset,shuffle=shuffle)
    return dataloader,sentences, bert_examples
    







class BertExample:
  def __init__(self):
    self.add_cls_sep = True
    self.sentence_id = -1
    self.orig_to_tok_index = []
    self.tok_to_orig_index = []
    self.techniques = None
    self.sentence = None
    self.labels = None
    self.context = None
    self.techniques = None
    self.tokens = []
    self.tokens_ids = []
    self.input_mask = []
  def __str__(self):
    print("sentence_id", self.sentence_id)
    return ""





def getEmbedding(tag: str, mode: str = "pos"):
    """
    Turn a single POS or NER tag string into a one-hot Python list.
    """
    if mode == "pos":
        idx = pos2id.get(tag)
        size = len(pos2id)
    elif mode == "ner":
        idx = ner2id.get(tag)
        size = len(ner2id)
    else:
        raise ValueError("mode must be 'pos' or 'ner'")

    if idx is None:
        # optional: handle unknown tags
        raise KeyError(f"Unknown {mode.upper()} tag: {tag}")

    vec = [0] * size
    vec[idx] = 1
    return torch.tensor(vec)


def convert_sentence_to_input_feature(sentence, sentence_id, add_cls_sep=True, max_seq_len=256):
  bert_example = BertExample()
  bert_example.sentence_id = sentence_id
  bert_example.add_cls_sep = add_cls_sep
  
  sentence_tokens = sentence.tokens
  sentence_labels = sentence.labels 
  sentence_techniques = sentence.techniques
  sentence_context = []
  for (i,j) in zip(sentence.pos,sentence.ner):
     pos =getEmbedding(i,mode = "pos")
     ner=getEmbedding(j, mode = "ner")
     sentence_context.append(torch.cat((pos,ner)))

    
  bert_example.tokens = sentence_tokens
  # index of which subtokens are part of which word

  tok_to_orig_index = sentence.tok2idx
  orig_to_tok_index = []
  
  bert_example.tok_to_orig_index = tok_to_orig_index
  bert_example.orig_to_tok_index = orig_to_tok_index

  #tok_to_orig_index =sentence.tok2idx
  
  
  
  tokens_ids = config.tokenizer.convert_tokens_to_ids(sentence_tokens)
  input_mask = [1] * len(tokens_ids)
  
  if sentence_labels is None:
    return bert_example
  all_doc_tokens = sentence_tokens
  labels = [0] * len(all_doc_tokens)
  for index, token in enumerate(all_doc_tokens):
    labels[index] = sentence_labels[index]
  
  assert len(labels) == len(tokens_ids)
  while len(labels) < max_seq_len:
    labels.append(-100)
  sentence_text = sentence.sentence
  assert(len(tokens_ids) == len(tok_to_orig_index))
  while len(tokens_ids) < max_seq_len:
    tokens_ids.append(0)
    input_mask.append(0)
    tok_to_orig_index.append(0)
 
  bert_example.tok_to_orig_index = tok_to_orig_index
  bert_example.tokens_ids = tokens_ids
  bert_example.input_mask = input_mask
  
  bert_example.sentence = sentence_text
  while len(sentence_techniques) < max_seq_len:
    sentence_techniques.append(-100)
    new_tensor = torch.empty_like(sentence_context[0])
    sentence_context.append(new_tensor)

  bert_example.techniques = sentence_techniques
  bert_example.labels = labels
  bert_example.context = torch.stack(sentence_context, dim=0)


  return bert_example 





def split_long_sentence(doc, max_len=256):
    """
    Yield sentence spans from the doc. If a sentence's tokenized length exceeds max_len,
    split it into consecutive sub-spans where each sub-span's tokenized length is <= max_len.
    
    Parameters:
        doc (spacy.tokens.Doc): The processed Doc.
        max_len (int): Maximum number of tokens per sentence chunk after tokenization.
    
    Yields:
        spacy.tokens.Span: A sentence or sub-sentence span with global character offsets.
    """
    for sent in doc.sents:
        sent_text = sent.text
        tokenized_sent = config.tokenizer.tokenize(sent_text)
        tokenized_sent = config.tokenizer.convert_tokens_to_ids(tokenized_sent)
        if len(tokenized_sent) <= max_len:
            yield sent
        else:
            sent_tokens = list(sent_text)
            start_idx = 0
            total_tokens = len(sent_tokens)
            
            while start_idx < total_tokens:
                low = start_idx
                high = total_tokens
                best_end = start_idx
                
                # Binary search to find the maximum end index where token count <= max_len
                while low < high:
                    mid = (low + high) // 2
                    if mid == start_idx:
                        chunk_text = ""
                    else:
                        chunk_start = sent_tokens[start_idx].idx
                        chunk_end = sent_tokens[mid - 1].idx + len(sent_tokens[mid - 1].text)
                        chunk_text = doc.text[chunk_start:chunk_end]
                    tokenized_chunk = config.tokenizer.tokenize(chunk_text)
                    length = len(config.tokenizer.convert_tokens_to_ids(tokenized_chunk))
                    if length <= max_len:
                        best_end = mid
                        low = mid + 1
                    else:
                        high = mid
                
                # Handle case where no valid split found (force move by 1 token)
                if best_end == start_idx:
                    best_end = start_idx + 1
                    if best_end > total_tokens:
                        best_end = total_tokens
                
                # Create span from character offsets
                chunk_start_char = sent_tokens[start_idx].idx
                chunk_end_char = sent_tokens[best_end - 1].idx + len(sent_tokens[best_end - 1]) if best_end > start_idx else sent_tokens[start_idx].end_char
                chunk_span = doc.char_span(chunk_start_char, chunk_end_char)
                
                if chunk_span is None:
                    # Fallback to token indices if char_span fails
                    chunk_span = doc[sent_tokens[start_idx].i : sent_tokens[best_end - 1].i + 1]
                
                yield chunk_span
                start_idx = best_end

def split_into_chunks(text, chunk_size=240):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

nlp = spacy.load("en_core_web_sm")

def get_example_bert_sentences(article, span=None, article_index=None,article_techniques= None):
  """
  Returns tokenized sentences in the BERT format
  """
  doc_tokens = [] # builds up to a word
  char_to_word_offset = [None] * len(article)
  all_sentence_tokens = []
  tokens_to_words = []
  word_to_start_char_offset = {}
  word_to_end_char_offset = {}
  all_pos_tags = []
  all_ner_tags = []
  doc = nlp(article)
  # we're iterating thorugh spacy sentences -> to accurately map spacy representations to bert/roberta ones we must know the length of each sentence
  prev_sentence_end = 0
  

  # in rare instances sentences can end up being longer than 256 words which causes issues.
  for x,sent in enumerate(split_long_sentence(doc, max_len=500)):
    sentence_tokens= config.tokenizer.tokenize(sent.text,add_special_tokens=True)
    encoded = config.tokenizer(
            sent.text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
    

    pos_tags = []
    ner_tags = []

    bert_offsets = encoded.offset_mapping  # list of (start, end)

    # 3) Build a list of spaCy token spans
    #    (start_char, end_char, index, text)
    
    spacy_spans = [
        (tok.idx, tok.idx + len(tok.text), i, tok.tag_, tok.ent_type_)
        for i, tok in enumerate(sent)
    ]

    # 4) For each BERT token, find the spaCy token(s) it overlaps
    alignment = []

    alignment.append(0)
    pos_tags.append("CLS")
    ner_tags.append("CLS")
    for i, ((b_start, b_end), b_tok) in enumerate(zip(bert_offsets, sentence_tokens[1:])):
        # find all spaCy tokens whose span overlaps [b_start, b_end)
        overlaps = [
            (i_sp, sp_pos,sp_ner)
            for (s_start, s_end, i_sp, sp_pos,sp_ner) in spacy_spans
            if not (s_end-prev_sentence_end <= b_start or s_start-prev_sentence_end >= b_end)
        ]
        # here we’ll just take the first overlap
        if overlaps:
            
            sp_idx, sp_pos,sp_ner = overlaps[0]
        else:
            sp_idx, sp_pos,sp_ner = 0, 'None', 'None'

        pos_tags.append( sp_pos
            )
        ner_tags.append(sp_ner)
        alignment.append(sp_idx
)
        # quick and dirty fix -> if token index is way off (greater than the number of subtokens) we set it back to earth.
    for i in range(0, len(alignment)):
       if alignment[i]>len(alignment):
          alignment[i]=len(alignment)
    alignment.append(len(alignment))
    prev_sentence_end=sent.end_char
    pos_tags.append("SEP")
    ner_tags.append("SEP")

    

    
    offset_mapping = encoded["offset_mapping"]
    all_sentence_tokens.append(sentence_tokens)
    all_pos_tags.append(pos_tags)
    all_ner_tags.append(ner_tags)
    tokens_to_words.append(alignment)
    prev_token_end = 0
    sent_start_char = sent.start_char
    for i, (token_start, token_end) in enumerate(offset_mapping):
            start = sent_start_char + token_start
            end = sent_start_char + token_end

            # Map token to global character offsets
            word_to_start_char_offset[(x, i)] = start
            word_to_end_char_offset[(x, i)] = end

            # Map characters to token positions
            for char_pos in range(start, end):
                if char_pos < len(article):
                    char_to_word_offset[char_pos] = (x, i+1)




  last_valid = None
  for i in range(len(char_to_word_offset)):
      current = char_to_word_offset[i]
      if current != None:
          last_valid = current
      else:
          if last_valid is not None:
              char_to_word_offset[i] = last_valid
  
  if span is None:
    return all_sentence_tokens, (word_to_start_char_offset, word_to_end_char_offset)

  current_propaganda_labels = []
  first_tokens = []
  current_technique_labels= []
  for doc_tokens in all_sentence_tokens:
    first_tokens.append([])
    for i in doc_tokens:
        if ((i[0] != "#" and config.LANGUAGE_MODEL=='BERT') or (i[0] == "Ġ" and config.LANGUAGE_MODEL=='RoBERTa')):
          first_tokens[-1].append(1)
        else:
          first_tokens[-1].append(0)
    current_propaganda_labels.append([0] * len(doc_tokens))
    current_technique_labels.append([0] * len(doc_tokens))


  start_positions = []
  end_positions = []
  techniques = []
  for i,sp in enumerate(span):
    
    if (char_to_word_offset[sp[0]][0] != char_to_word_offset[sp[1]-1][0]):
      # if the sentence of the character at the beggining of the span isn't the same as sentence of the character at the end of the span.
      l1 = char_to_word_offset[sp[0]][0] # sentence 1
      l2 = char_to_word_offset[sp[1] - 1][0] # sentence 2
      start_positions.append(char_to_word_offset[sp[0]]) 
      techniques.append(article_techniques[i]) 
      # add (sentence, index of curr word) of starting word for span.
      end_positions.append((l1, len(all_sentence_tokens[l1])-1))
      # add (sentence 1, lenth of sentence 1) i.e say that the span goes to the end of the sentence.
      l1 += 1
      while(l1 < l2):
        start_positions.append((l1, 0)) # add another span starting at next sentence from sentence 1
        techniques.append(article_techniques[i])
        end_positions.append((l1, len(all_sentence_tokens[l1])-1))
        l1 += 1
        # keep filling sentences with spans until sentence 2 reached.
      start_positions.append((l2, 0)) 
      techniques.append(article_techniques[i])
      end_positions.append(char_to_word_offset[sp[1]-1])  
      continue
    # add whichever character is at the outlined char
    start_positions.append(char_to_word_offset[sp[0]])
    end_positions.append(char_to_word_offset[sp[1]-1])
    techniques.append(article_techniques[i])
  for i, _ in enumerate(start_positions):
    assert start_positions[i][0] == end_positions[i][0]
    current_propaganda_labels[start_positions[i][0]][start_positions[i][1]] = 2 # Begin label
    if techniques != None:
        try:
          current_technique_labels[start_positions[i][0]][start_positions[i][1]] = tag2idx[techniques[i]]
        except:
           pass

    if start_positions[i][1] < end_positions[i][1]:
        for loc in range(start_positions[i][1] + 1,end_positions[i][1] + 1):
          if first_tokens[start_positions[i][0]][loc] == 1:
            current_propaganda_labels[start_positions[i][0]][loc] = 1
          else:
            current_propaganda_labels[start_positions[i][0]][loc] = 1
            #current_propaganda_labels[start_positions[i][0]][loc] = -100
        try:
          current_technique_labels[start_positions[i][0]][start_positions[i][1]+ 1 : end_positions[i][1] + 1] = [tag2idx[techniques[i]]] * (end_positions[i][1] - start_positions[i][1])
        except:
           pass
      
  num_sentences = len(all_sentence_tokens)
  start_offset_list = get_list_from_dict(num_sentences, word_to_start_char_offset)
  end_offset_list = get_list_from_dict(num_sentences, word_to_end_char_offset)
  output_sentences = []
  for i in range(num_sentences):
    sentence = ExampleSentence  ()
    sentence.tokens = all_sentence_tokens[i]
    sentence.tok2idx = tokens_to_words[i]
    sentence.ner = all_ner_tags[i]
    sentence.pos = all_pos_tags[i]
    sentence.length = len(all_sentence_tokens[i])
    sentence.techniques = current_technique_labels[i]
    sentence.labels = current_propaganda_labels[i]
    sentence.article_index =  article_index
    sentence.index = i
    sentence.word_to_start_char_offset = start_offset_list[i]
    sentence.word_to_end_char_offset = end_offset_list[i]
    num_words = len(sentence.tokens)
    try:
      assert len(sentence.labels) == num_words == len(sentence.ner) == len(sentence.pos) == len(sentence.tok2idx)
    except:
       print(len(sentence.labels))
       print(num_words)
       print(sentence.tokens)
       print(sentence.ner)
       print(sentence.pos)
       print(len(sentence.ner))
       print(len(sentence.pos))
       print(len(sentence.tok2idx))
       assert False
    assert len(sentence.word_to_start_char_offset) == num_words-2
    assert len(sentence.word_to_end_char_offset) == num_words-2
    output_sentences.append(sentence)

  return all_sentence_tokens, current_propaganda_labels, (word_to_start_char_offset, word_to_end_char_offset), output_sentences

def get_example_bert_sentences_manual(article, span=None, article_index=None,article_techniques= None):
  """
  Returns tokenized sentences in the BERT format
  """
  doc_tokens = [] # builds up to a word
  char_to_word_offset = [None] * len(article)
  all_sentence_tokens = []
  tokens_to_words = []
  word_to_start_char_offset = {}
  word_to_end_char_offset = {}
  all_pos_tags = []
  all_ner_tags = []
  doc = nlp(article)
  # we're iterating thorugh spacy sentences -> to accurately map spacy representations to bert/roberta ones we must know the length of each sentence
  prev_sentence_end = 0
  

  # in rare instances sentences can end up being longer than 256 words which causes issues.
  for index, c in enumerate(article):
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
      if prev_is_whitespace:
        doc_tokens.append(c)
        current_word_position = (len(all_sentence_tokens), len(doc_tokens) - 1)
        word_to_start_char_offset[current_word_position] = index # start offset of word
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append((len(all_sentence_tokens), len(doc_tokens) - 1))
  if doc_tokens:
    all_sentence_tokens.append(doc_tokens)
  if current_word_position is not None:
    word_to_end_char_offset[current_word_position] = index
    current_word_position = None




  last_valid = None
  for i in range(len(char_to_word_offset)):
      current = char_to_word_offset[i]
      if current != None:
          last_valid = current
      else:
          if last_valid is not None:
              char_to_word_offset[i] = last_valid
  
  if span is None:
    return all_sentence_tokens, (word_to_start_char_offset, word_to_end_char_offset)

  current_propaganda_labels = []
  first_tokens = []
  current_technique_labels= []
  for doc_tokens in all_sentence_tokens:
    first_tokens.append([])
    for i in doc_tokens:
        if ((i[0] != "#" and config.LANGUAGE_MODEL=='BERT') or (i[0] == "Ġ" and config.LANGUAGE_MODEL=='RoBERTa')):
          first_tokens[-1].append(1)
        else:
          first_tokens[-1].append(0)
    current_propaganda_labels.append([0] * len(doc_tokens))
    current_technique_labels.append([0] * len(doc_tokens))


  start_positions = []
  end_positions = []
  techniques = []
  for i,sp in enumerate(span):
    
    if (char_to_word_offset[sp[0]][0] != char_to_word_offset[sp[1]-1][0]):
      # if the sentence of the character at the beggining of the span isn't the same as sentence of the character at the end of the span.
      l1 = char_to_word_offset[sp[0]][0] # sentence 1
      l2 = char_to_word_offset[sp[1] - 1][0] # sentence 2
      start_positions.append(char_to_word_offset[sp[0]]) 
      techniques.append(article_techniques[i]) 
      # add (sentence, index of curr word) of starting word for span.
      end_positions.append((l1, len(all_sentence_tokens[l1])-1))
      # add (sentence 1, lenth of sentence 1) i.e say that the span goes to the end of the sentence.
      l1 += 1
      while(l1 < l2):
        start_positions.append((l1, 0)) # add another span starting at next sentence from sentence 1
        techniques.append(article_techniques[i])
        end_positions.append((l1, len(all_sentence_tokens[l1])-1))
        l1 += 1
        # keep filling sentences with spans until sentence 2 reached.
      start_positions.append((l2, 0)) 
      techniques.append(article_techniques[i])
      end_positions.append(char_to_word_offset[sp[1]-1])  
      continue
    # add whichever character is at the outlined char
    start_positions.append(char_to_word_offset[sp[0]])
    end_positions.append(char_to_word_offset[sp[1]-1])
    techniques.append(article_techniques[i])
  for i, _ in enumerate(start_positions):
    assert start_positions[i][0] == end_positions[i][0]
    current_propaganda_labels[start_positions[i][0]][start_positions[i][1]] = 2 # Begin label
    if techniques != None:
        try:
          current_technique_labels[start_positions[i][0]][start_positions[i][1]] = tag2idx[techniques[i]]
        except:
           pass

    if start_positions[i][1] < end_positions[i][1]:
        for loc in range(start_positions[i][1] + 1,end_positions[i][1] + 1):
          if first_tokens[start_positions[i][0]][loc] == 1:
            current_propaganda_labels[start_positions[i][0]][loc] = 1
          else:
            current_propaganda_labels[start_positions[i][0]][loc] = 1
            #current_propaganda_labels[start_positions[i][0]][loc] = -100
        try:
          current_technique_labels[start_positions[i][0]][start_positions[i][1]+ 1 : end_positions[i][1] + 1] = [tag2idx[techniques[i]]] * (end_positions[i][1] - start_positions[i][1])
        except:
           pass
      
  num_sentences = len(all_sentence_tokens)
  start_offset_list = get_list_from_dict(num_sentences, word_to_start_char_offset)
  end_offset_list = get_list_from_dict(num_sentences, word_to_end_char_offset)
  output_sentences = []
  for i in range(num_sentences):
    sentence = ExampleSentence  ()
    sentence.tokens = all_sentence_tokens[i]
    sentence.tok2idx = tokens_to_words[i]
    sentence.ner = all_ner_tags[i]
    sentence.pos = all_pos_tags[i]
    sentence.length = len(all_sentence_tokens[i])
    sentence.techniques = current_technique_labels[i]
    sentence.labels = current_propaganda_labels[i]
    sentence.article_index =  article_index
    sentence.index = i
    sentence.word_to_start_char_offset = start_offset_list[i]
    sentence.word_to_end_char_offset = end_offset_list[i]
    num_words = len(sentence.tokens)
    try:
      assert len(sentence.labels) == num_words == len(sentence.ner) == len(sentence.pos) == len(sentence.tok2idx)
    except:
       print(len(sentence.labels))
       print(num_words)
       print(sentence.tokens)
       print(sentence.ner)
       print(sentence.pos)
       print(len(sentence.ner))
       print(len(sentence.pos))
       print(len(sentence.tok2idx))
       assert False
    assert len(sentence.word_to_start_char_offset) == num_words-2
    assert len(sentence.word_to_end_char_offset) == num_words-2
    output_sentences.append(sentence)

  return all_sentence_tokens, current_propaganda_labels, (word_to_start_char_offset, word_to_end_char_offset), output_sentences





def is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    return True
  return False


def get_list_from_dict(num_sentences, word_offsets):
  li = []
  for _ in range(num_sentences):
    li.append([])
  for key in word_offsets:
    si = key[0]
    li[si].append(word_offsets[key])

  return li


def get_tokens(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence) # 
        
    # return text as a list of tokens
    return [(token.text,token.idx, token.idx + len(token.text)) for token in doc]



def get_dataloader(inputs,labels,context,masks,sentence_ids,techniques,tok2idx,return_dataset=False,shuffle='error'):
    tensor_data = TensorDataset(inputs, labels,context, masks, sentence_ids,techniques,tok2idx)
    dataloader = DataLoader(tensor_data, batch_size=config.BATCH_SIZE,shuffle=shuffle)
    if return_dataset:
       return dataloader, tensor_data
    return dataloader

def get_packed_sequence(representations, labels):
    input_lengths = []
    # padded_representations = [
    # F.pad(sentence, (0, 0, 0, 256 - sentence.shape[0]))
    # for sentence in representations
    # ]
    packed_sequence = pack_sequence(representations,enforce_sorted=False)
    print(f"Packed batch lengths: {packed_sequence.batch_sizes}")
    return packed_sequence, labels

def collate_fn(batch):
    """Collate function to pad sequences and compute lengths."""
    representations, labels,techniques = zip(*batch)
    
    # Compute sequence lengths before padding
    lengths = torch.tensor([rep.shape[0] for rep in representations])
    labels = [torch.tensor(label) for label in labels]
    techniques = [torch.tensor(technique) for technique in techniques]
    # Pad sequences to the maximum length in the batch
    padded_representations = pad_sequence(representations, batch_first=True, padding_value=0.0)
    # Pad labels (if needed)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # Use -1 for ignored labels
    padded_techniques = pad_sequence(techniques,batch_first=True,padding_value=-1)
    # stack labels
    assert padded_representations.shape[0] == padded_labels.shape[0]
    assert padded_representations.shape[1] == padded_labels.shape[1]

    return padded_representations, lengths, padded_labels, padded_techniques

def get_padded_dataloader(sentences,batch_size=config.BATCH_SIZE):
    """Returns a DataLoader that provides padded sequences."""
    representations = [sentence.tokens_ids for sentence in sentences]
    labels = [sentence.labels for sentence in sentences]
    techniques = [sentence.techniques for sentence in sentences]
    if techniques == None:
      dataset = list(zip(representations, labels))
    else:
       dataset= list(zip(representations,labels,techniques
       ))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader
