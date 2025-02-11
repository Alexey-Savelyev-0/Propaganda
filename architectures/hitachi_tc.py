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
    
class FFN_LABEL_WISE(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=128):
        super(FFN_LABEL_WISE, self).__init__()
        
        # Create separate FFN layers for each label
        self.label_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # Output a single logit per label
            ) for _ in range(num_labels)
        ])
    
    def forward(self, x):
        # Apply each label's FFN separately and concatenate results
        outputs = [ffn(x).squeeze(-1) for ffn in self.label_ffns]
        return torch.stack(outputs, dim=1)  # Shape: (batch_size, num_labels)
    



class HITACHI_TC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HITACHI_TC, self).__init__()
        self.FFN_BOS = FFN_BOS(input_dim, hidden_dim, output_dim)
        self.FFN_SPAN = FFN_SPAN(input_dim, hidden_dim, output_dim)
        self.FFN_LABEL_WISE = FFN_LABEL_WISE(input_dim, output_dim)
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        """The propaganda span representation is obtained by concatenating the representation of the
        BoS token (e(tc)  BoS), tokens located at span start  (e(tc)  start) and end (e(tc)  end), 
        and representations aggregated by attention (e(tc)  att ) and maxpooling (e(mtca)xp)"""


        #only parse BoS tokens into the first FFN, and the rest into the second FFN
        bos_token = x[:, 0, :]  # Assuming the first token in the sequence is the BoS token
        span_tokens = x[:, 1:, :]  # The rest are span tokens

        bos_output = self.FFN_BOS(bos_token)
        span_output = self.FFN_SPAN(span_tokens)
        e_tc_span = bos_output + span_output
        # simgoid activation
        # Define learnable parameters for each class
        v_tc_l = nn.Parameter(torch.randn(self.output_dim))
        b_tc_l = nn.Parameter(torch.randn(self.output_dim))

        # Apply the label-wise FFN and compute the final output
        y_tc_l = torch.sigmoid(v_tc_l * self.FFN_LABEL_WISE(e_tc_span) + b_tc_l)
        


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

    
    hidden_states = torch.stack(hidden_states, dim=0)

    
    s = nn.Parameter(torch.randn(num_layers))  # Attention weights
    c = nn.Parameter(torch.ones(1))  # Scaling factor

    attn_weights = F.softmax(s, dim=-1)  # Shape: (batch_size, seq_len, num_layers)

    
    attn_weights = attn_weights.permute(2, 0, 1).unsqueeze(-1)  

    fused_embeddings = torch.sum(attn_weights * hidden_states, dim=0)  

    return c * fused_embeddings  # Apply scaling factor

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


def hitachi_tc_train():
    hitachi_si = HITACHI_SI()
    articles, article_ids = classification.read_articles("train-articles")
    spans, techniques = classification.read_spans()
    NUM_ARTICLES = 10
    articles = articles[0:NUM_ARTICLES]
    spans = spans[0:NUM_ARTICLES]
    techniques = techniques[0:NUM_ARTICLES]
    indices = np.arange(NUM_ARTICLES)
    train_articles, eval_articles, train_spans, eval_spans, train_techniques, eval_techniques, train_indices, eval_indices = train_test_split(articles, spans, techniques, indices, test_size=0.2)
    train_dataloader = get_data_hitachi(train_articles, train_spans, train_techniques)
    eval_dataloader = get_data_hitachi(eval_articles, eval_spans, eval_techniques)
    model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    epochs = 1
    for epoch_i in range(0, epochs):
        hitachi_si.train()
        for step, batch in enumerate(train_dataloader):
            print(batch[1].shape)
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_lengths = batch[3].to(device)
            with torch.no_grad():        
                outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
                loss = outputs[0]

                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                scheduler.step() # TODO


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