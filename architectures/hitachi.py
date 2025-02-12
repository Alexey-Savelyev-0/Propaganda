"""Propaganda Span Representation: To produce a propaganda
 span representation, we provide two  distinct FFNs, 
feeding input representation h(tc)  i , that were obtained in
 the same manner as the SI model. One of the two FFNs is
 for the BoS token and produces sentence representations, and 
 the other is for tokens in a propaganda span:"""

"""Input representation:"""

""" propaganda span representation is obtained by 
concatienag the representation of the Bos token, span start and end tokens,
and repersentations aggregated by attention and maxpooling in the span as follows."""

"""We provide an additional label-wise FFN and linear layer to extract label-pecific 
information for each propaganda technique before prediction."""


"""
Input Representation: To obtain input representations, we provide a 
layer-wise attention to fuse the outputs of PLM layers 
(Kondratyuk and Straka, 2019; Peters et al., 2018):
"""



import torch
from torch import nn
import spacy
import numpy as np
import torch.nn.functional as F
# 2 FFNs

class FFN_BOS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN_BOS, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

class FFN_SPAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN_SPAN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
    
def get_embedding(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    return doc.vector

def get_plm_representation(sentence, tokenizer, model, add_cls_sep=True, max_seq_len=150):
    tokenized_sentence = tokenizer.encode_plus(sentence,
                                             add_special_tokens=add_cls_sep,
                                             max_length=max_seq_len,
                                             pad_to_max_length=True,
                                             return_attention_mask=True)
    return tokenized_sentence['input_ids'], tokenized_sentence['attention_mask']

def get_PLM_layer_attention(sentence, model, tokenizer):
    """ To obtain input representations, we provide a layer-wise attention to fuse the outputs of PLM layers.
    To obtain the ith word, we sum PLM(i,j) over j, j being the layer index. In this sum
    we apply a softmax to a trainable parameter vector, which is multiplied by the output of the PLM layer.
    The concrete details may be found in the paper.
    """
    
    inputs = get_plm_representation(sentence, tokenizer, model)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # Tuple of shape (num_layers, batch_size, seq_len, hidden_dim)

    num_layers = len(hidden_states)
    batch_size, seq_len, hidden_dim = hidden_states[0].shape

     # Convert hidden_states tuple to a tensor
    hidden_states = torch.stack(hidden_states, dim=0)

    # h = vector c multiplied by sum of jth layer of hidden states
    s = nn.Parameter(torch.randn(num_layers))  # Attention weights
    c = nn.Parameter(torch.ones(1))  # Scaling factor
    
    output = torch.zeros(hidden_states.shape[1], hidden_states.shape[2])
    h = torch.zeros(hidden_states.shape[1], hidden_states.shape[2])
    for i in range(hidden_states.shape[1]):
        for j in range(hidden_states.shape[0]):
            h = h + softmax(s) * hidden_states[i][j]
        h = h * c
        output[i] = h
    return output

def get_special_tags(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    pos_tags = list(nlp.get_pipe("tok2vec").labels)  # Get all possible POS tags
    ner_tags = list(nlp.get_pipe("ner").labels)  # Get all possible entity labels
    word_embeddings = []
    for token in doc:
        # Contextual embedding from transformer model

        # One-hot encoding for PoS
        pos_one_hot = np.zeros(len(pos_tags))
        if token.pos_ in pos_tags:
            pos_one_hot[pos_tags.index(token.pos_)] = 1

        # One-hot encoding for Named Entity (NER)
        ner_one_hot = np.zeros(len(ner_tags))
        if token.ent_type_ in ner_tags:
            ner_one_hot[ner_tags.index(token.ent_type_)] = 1

        # Concatenate embeddings
        combined_embedding = np.concatenate([pos_one_hot, ner_one_hot])
        word_embeddings.append(combined_embedding)
    return word_embeddings

def get_token_representation(sentence):
    """Token representation is obtained by concatting the plm representation, the PoS tag, and NE tag."""
    plm = get_PLM_layer_attention(sentence)
    ner, pos = get_special_tags(sentence)
    # if plm layer has multiple tokens for one word, we can average them

    return torch.cat((plm, ner, pos))





nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")


word_embeddings = []
pos_tags = list(nlp.get_pipe("tok2vec").labels)  # Get all possible POS tags
ner_tags = list(nlp.get_pipe("ner").labels)  # Get all possible entity labels

for token in doc:
    # Contextual embedding from transformer model
    
    # One-hot encoding for PoS
    pos_one_hot = np.zeros(len(pos_tags))
    if token.pos_ in pos_tags:

        pos_one_hot[pos_tags.index(token.pos_)] = 1
    
    # One-hot encoding for Named Entity (NER)
    ner_one_hot = np.zeros(len(ner_tags))
    if token.ent_type_ in ner_tags:
        ner_one_hot[ner_tags.index(token.ent_type_)] = 1

    # Concatenate embeddings
    combined_embedding = np.concatenate([ pos_one_hot, ner_one_hot])
    word_embeddings.append(combined_embedding)
print(word_embeddings)