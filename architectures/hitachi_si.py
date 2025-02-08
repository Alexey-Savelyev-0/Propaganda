import transformers
import torch
from torch import nn
import numpy as np
import spacy
# import crf
import torch.nn.functional as F
from torchcrf import CRF




class HITACHI_SI(nn.Module):
    def __init__(self,PLM="BERT", input_dim=None, hidden_dim=None, output_dim=None):
        super(HITACHI_SI, self).__init__()
        if PLM == "BERT":
            self.PLM = transformers.BertModel.from_pretrained('bert-base-uncased')

        self.BILSTM = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.output_dim = output_dim
        self.linear = nn.Linear(hidden_dim*2, output_dim)
        self.relu = nn.ReLU()
        # change
        self.FFN = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.CRF = CRF(output_dim)

    def forward(self, x):
        pass






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




if __name__ == "__main__":
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