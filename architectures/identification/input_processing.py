import torch
from .hitachi_utils import config
import spacy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_sequence
import torch.nn.functional as F
from transformers import BertTokenizerFast

device = config.device

class example_sentence:
  def __init__(self):
    self.tokens = []
    self.labels = []
    self.techniques = []
    self.sentence = None
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


def get_data(articles, spans, indices,techniques=None):
  assert len(articles) == len(spans)    
  sentences = []
  for index in indices:
    article = articles[index]
    span = spans[index]
    article_techniques = techniques[index] if techniques != None else None
    _, _, _, cur_sentences = get_sentence_tokens_labels(article, span, index, article_techniques)
    sentences += cur_sentences
    #sentences[list[Sentence]]
  
  
  
  bert_examples = []

  for i, sentence in enumerate(sentences):
    if len(sentence.tokens) > 256:
      print("Sentence length greater than 256 aa")
      print(len(sentence.tokens))
    input_feature = convert_sentence_to_input_feature(sentence, i, config.tokenizer)
    bert_examples.append(input_feature)
  dataloader = get_dataloader(bert_examples)
  
  return dataloader, sentences, bert_examples


class BertExample:
  def __init__(self):
    self.add_cls_sep = True
    self.sentence_id = -1
    self.orig_to_tok_index = []
    self.tok_to_orig_index = []
    self.techniques = None
    self.sentence = None
    self.labels = None
    self.techniques = None
    self.tokens = []
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
  sentence_techniques = sentence.techniques
  # index of which subtokens are part of which word
  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = [] 
  for (i, token) in enumerate(sentence_tokens):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenizer.tokenize(token)
    tok_to_orig_index.append(i)
    all_doc_tokens.append(token)
  bert_example.tok_to_orig_index = tok_to_orig_index
  bert_example.orig_to_tok_index = orig_to_tok_index
  
  bert_tokens = all_doc_tokens
  
  tokens_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
  input_mask = [1] * len(tokens_ids)
  
  # bert_example.input_mask = [float(i>0) for i in token_ids]

  if sentence_labels is None:
    return bert_example
  
  # we only care about first tokens of word
  labels = [0] * len(all_doc_tokens)
  # the label of each word is replicated for each subtoken of the word
  # make sure to at some point allow only the inclusion of the first token of the word
  
  for index, token in enumerate(all_doc_tokens):
    labels[index] = sentence_labels[tok_to_orig_index[index]]
  
  assert len(labels) == len(tokens_ids)
  while len(labels) < max_seq_len:
    labels.append(-100)
  sentence_text = sentence.sentence
  
  while len(tokens_ids) < max_seq_len:
    tokens_ids.append(1)
    input_mask.append(0)
  bert_example.tokens_ids = tokens_ids
  bert_example.input_mask = input_mask
  
  bert_example.sentence = sentence_text
  while len(sentence_techniques) < max_seq_len:
    sentence_techniques.append(0)
  bert_example.techniques = sentence_techniques
  bert_example.labels = labels

  return bert_example 


def split_long_sentence(doc, max_len=256):
    """
    Yield sentence spans from the doc. If a sentence has more than max_len tokens,
    split it into consecutive sub-spans of at most max_len tokens.
    
    Parameters:
        doc (spacy.tokens.Doc): The processed Doc.
        max_len (int): Maximum number of tokens per sentence chunk.
    
    Yields:
        spacy.tokens.Span: A sentence or sub-sentence span with global character offsets.
    """
    for sent in doc.sents:
        sentence_tokens= config.tokenizer.tokenize(sent.text)
        if len(sentence_tokens) <= max_len:
            yield sent
        else:
            # If the sentence is too long, split it into chunks
            for i in range(0, len(sent), max_len):
                # Calculate absolute start and end positions
                start = sent.start + i
                end = min(sent.start + i + max_len, sent.end)
                yield doc[start:end]


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

def get_sentence_tokens_labels(article, span=None, article_index=None,article_techniques = None):
  doc_tokens = [] # builds up to a word
  char_to_word_offset = [None] * len(article)
  all_sentence_tokens = []
  word_to_start_char_offset = {}
  word_to_end_char_offset = {}
  nlp = spacy.load("en_core_web_sm")
  doc = nlp(article)
  
  for x,sent in enumerate(split_long_sentence(doc, max_len=240)):
    sentence_tokens= config.tokenizer.tokenize(sent.text,add_special_tokens=True)
    encoded = config.tokenizer(
            sent.text,
            return_offsets_mapping=True,
            add_special_tokens=False # Exclude <s>, </s> for alignment
        )
    offset_mapping = encoded["offset_mapping"]
    all_sentence_tokens.append(sentence_tokens)
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
  current_technique_labels= []
  for doc_tokens in all_sentence_tokens:
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
    if config.TAGGING_SCHEME == "BIO":
      current_propaganda_labels[start_positions[i][0]][start_positions[i][1]] = 2 # Begin label
      if techniques != None:
        current_technique_labels[start_positions[i][0]][start_positions[i][1]] = techniques[i]

      if start_positions[i][1] < end_positions[i][1]:
        current_propaganda_labels[start_positions[i][0]][start_positions[i][1] + 1 : end_positions[i][1] + 1] = [1] * (end_positions[i][1] - start_positions[i][1])
      
  num_sentences = len(all_sentence_tokens)

  start_offset_list = get_list_from_dict(num_sentences, word_to_start_char_offset)
  end_offset_list = get_list_from_dict(num_sentences, word_to_end_char_offset)
  output_sentences = []
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


def get_PLM_layer_attention(sentence, model,c, s,tokenizer,avg_subtokens = True):
    """ To obtain input representations, we provide a layer-wise attention to fuse the outputs of PLM layers.
    To obtain the ith word, we sum PLM(i,j) over j, j being the layer index. In this sum
    we apply a softmax to a trainable parameter vector, which is multiplied by the output of the PLM layer.
    The concrete details may be found in the paper.
    """
    
    # for BERT
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    offset = inputs["offset_mapping"].tolist()[0]
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    spacy_sentence = get_tokens(sentence)
    num_words = len(spacy_sentence) + 2
    token_to_word_mapping = {}
    # Only works for BERT-like models
    """
    token_to_word_mapping[0]= cur_word
    for i in range(0,len(tokenizer_words)):
        subword = tokenizer_words[i]
        if subword[0]!="'" and( len(subword)<2 or  subword[:2] != "##"):
            cur_word+=1
        
        token_to_word_mapping[i+1] = cur_word
    token_to_word_mapping[len(tokenizer_words)+1] = cur_word+1
    """
    # ensure token_to_word_mapping same format as before
    token_to_word_mapping[0] = 0
    for sub_offset in offset[1:-1]:  # Skip CLS and SEP
        sub_start, sub_end = sub_offset
        mapped_token_index = None  # Default if no match is found

        # Check which token the subtoken belongs to
        for i, (_, token_start, token_end) in enumerate(spacy_sentence):
            # and sub_end<= token_end, but is it necessary?
            if sub_start >= token_start:
                mapped_token_index = i
                break    
        token_to_word_mapping[len(token_to_word_mapping)] = mapped_token_index
    token_to_word_mapping[len(token_to_word_mapping)] = len(spacy_sentence) - 1
    # currently BERT is frozen
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key,value in inputs.items()}
    with torch.no_grad():
        
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # Tuple of shape (num_layers, batch_size, seq_len, hidden_dim)

    if inputs["input_ids"].shape[1] +2 != num_words:
        # average the hidden states of (a,b,c,d) and (e,f,g,h) if same batch size i.e b==e and c and g belong to the same token
        pass

    num_layers = len(hidden_states)
    batch_size, seq_len, hidden_dim = hidden_states[0].shape

    hidden_states = torch.stack(hidden_states, dim=0)

    if avg_subtokens == True:
        averaged_hidden_states = torch.zeros((num_layers, batch_size, num_words, hidden_dim), device=hidden_states.device)
        
        word_indices = torch.tensor(list(token_to_word_mapping.values()), device=hidden_states.device)
        word_counts = torch.zeros((batch_size, num_words, 1), device=hidden_states.device)
        #averaged_hidden_states.scatter_add_(2, word_indices.view(1, 1, -1, 1).expand(num_layers, batch_size, -1, hidden_dim), hidden_states)
        averaged_hidden_states = averaged_hidden_states.scatter_add(2, word_indices.view(1, 1, -1, 1).expand(num_layers, batch_size, -1, hidden_dim), hidden_states)
        ones = torch.ones((batch_size, seq_len, 1), device=hidden_states.device)
        word_counts = word_counts.scatter_add(1, word_indices.view(1, -1, 1).expand(batch_size, -1, 1), ones)
        word_counts[word_counts == 0] = 1 
        averaged_hidden_states /= word_counts.unsqueeze(0)
        hidden_states=averaged_hidden_states


    # lets say for a word i we will have a PLM vector
    #s = nn.Parameter(torch.randn(num_layers))  # Attention weights
    # c is a scalar, s is a vector of dim num_layers 
    #c = nn.Parameter(torch.ones(1))  # Scaling factor

    attn_weights = F.softmax(s, dim=-1) 
    attn_weights = attn_weights.view(num_layers, 1, 1, 1)
    
    
    # for every token i, we go through layers j, multiplyingsoftmax(s) 
    fused_embeddings = torch.sum(attn_weights * hidden_states, dim=0)
    # I don't know why this is required
    test = c.clone()
    output = torch.mul(fused_embeddings, test)
    return fused_embeddings  # Apply scaling factor


    

def get_tokens(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence) # 
        
    # return text as a list of tokens
    return [(token.text,token.idx, token.idx + len(token.text)) for token in doc]

def get_special_tags(sentence):
    """ Tokenizes sentences, returns their special tags in dimensions (sentences, sentence-length, vector length)
    In Hitachi, special tags are: Part of Speech - PoS and Named Entities: One hot vector embeddings are generated for both
    If BERT is being used, 'special embeddings for CLS and SEP' are used.
    """


    nlp = spacy.load("en_core_web_sm")
    
    pos_tags = list(nlp.get_pipe("tagger").label_data)  
    ner_tags = list(nlp.get_pipe("ner").labels)  
    doc = nlp(sentence)
    batch_pos_embeddings = []
    batch_ner_embeddings = []
    pos_embeddings = []
    ner_embeddings = []
    pos_CLS,pos_SEP = torch.zeros(len(pos_tags)+2),torch.zeros(len(pos_tags)+2)
    pos_CLS[-1], pos_SEP[-2] = 1,1

    ner_CLS,ner_SEP = torch.zeros(len(ner_tags)+2),torch.zeros(len(ner_tags)+2)
    ner_CLS[-1], ner_SEP[-2] = 1,1

    pos_embeddings.append(pos_CLS)
    ner_embeddings.append(ner_CLS)
    

    for token in doc:
        # One-hot encoding for PoS, len+2 to account for CLS & SEP
        pos_one_hot = torch.zeros(len(pos_tags)+2)
        # part of speech is returning empty
        if token.pos_ in pos_tags:
            pos_one_hot[pos_tags.index(token.pos_)] = 1
        # One-hot encoding for Named Entity (NER)
        ner_one_hot = torch.zeros(len(ner_tags)+2)
        if token.ent_type_ in ner_tags:
            ner_one_hot[ner_tags.index(token.ent_type_)] = 1
        pos_embeddings.append(pos_one_hot)
        ner_embeddings.append(ner_one_hot)
    
    pos_embeddings.append(pos_SEP)
    ner_embeddings.append(ner_SEP)
    batch_pos_embeddings.append(torch.stack(pos_embeddings))
    batch_ner_embeddings.append(torch.stack(ner_embeddings))
    batch_pos_embeddings = torch.stack(batch_pos_embeddings)  
    batch_ner_embeddings = torch.stack(batch_ner_embeddings)  
    return batch_pos_embeddings, batch_ner_embeddings

def get_token_representation(sentence:list[str],model,s,c,tokenizer = config.tokenizer):
    ### assume sentence is already tokenized
    """Token representation is obtained by concatting the plm representation, the PoS tag, and NE tag."""
    plm = get_PLM_layer_attention(sentence,model=model,tokenizer=tokenizer, s=s, c=c)
    
    ner, pos = get_special_tags(sentence)
    plm = plm.to(device)
    ner = ner.to(device)
    pos = pos.to(device)
    output = torch.cat((plm, ner, pos),dim=-1)
    # flatten out to 2d, as first dimension is always 1
    
    assert output.shape[0] == 1
    output = output.squeeze(0)
    return output

def get_dataloader(examples):
    
        
    inputs = torch.tensor([d.tokens_ids for d in examples])
    labels = torch.tensor([d.labels for d in examples])
    masks = torch.tensor([d.input_mask for d in examples])

    sentence_ids = torch.tensor([d.sentence_id for d in examples])
    techniques = torch.tensor([d.techniques for d in examples])

    #padded_representations = [F.pad(sentence, (0, 0, 0, 256 - sentence.shape[0]))for sentence in inputs]
    #representations = torch.stack(examples,dim=0)
    #labels = torch.tensor(labels)
    tensor_data = TensorDataset(inputs, labels, masks, sentence_ids,techniques)
    dataloader = DataLoader(tensor_data, batch_size=config.BATCH_SIZE)
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader
