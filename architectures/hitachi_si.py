import transformers
import torch
from torch import nn
import numpy as np
import spacy
# import crf
import torch.nn.functional as F
from torchcrf import CRF


NLP = spacy.load("en_core_web_sm")


"""
To Do: 
1. Flesh out HITACHI_SI
2. Integrate with boilerplate code
3. Create auxillary tasks
The tasks are as follows:
1. Token - Level Technique Classification
2. Sentence Level Binary Classification
"""


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
    
    # test that this behaves in the way you expect it to.
    # at least dimensions are what they should be
    def get_PLM_layer_attention(self,sentence, model,tokenizer,avg_subtokens = True):
        """ To obtain input representations, we provide a layer-wise attention to fuse the outputs of PLM layers.
        To obtain the ith word, we sum PLM(i,j) over j, j being the layer index. In this sum
        we apply a softmax to a trainable parameter vector, which is multiplied by the output of the PLM layer.
        The concrete details may be found in the paper.
        """
        
        # for BERT
        inputs = tokenizer(sentence, return_tensors="pt")
        tokenizer_words = tokenizer.tokenize(sentence[0])
        num_words = len(self.get_embedding(sentence)) +2
        token_to_word_mapping = {}
        cur_word = 0
        # Only works for BERT-like models
        token_to_word_mapping[0]= cur_word
        for i in range(0,len(tokenizer_words)):
            subword = tokenizer_words[i]
            if len(subword)<2 or  subword[:2] != "##":
                cur_word+=1
            
            token_to_word_mapping[i+1] = cur_word
        token_to_word_mapping[len(tokenizer_words)+1] = cur_word+1
            

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
            averaged_hidden_states.scatter_add_(2, word_indices.view(1, 1, -1, 1).expand(num_layers, batch_size, -1, hidden_dim), hidden_states)
            ones = torch.ones((batch_size, seq_len, 1), device=hidden_states.device)
            word_counts.scatter_add_(1, word_indices.view(1, -1, 1).expand(batch_size, -1, 1), ones)
            word_counts[word_counts == 0] = 1 
            averaged_hidden_states /= word_counts.unsqueeze(0)
            hidden_states=averaged_hidden_states


        # lets say for a word i we will have a PLM vector
        s = nn.Parameter(torch.randn(num_layers))  # Attention weights
        # c is a scalar, s is a vector of dim num_layers 
        c = nn.Parameter(torch.ones(1))  # Scaling factor

        attn_weights = F.softmax(s, dim=-1) 
        attn_weights = attn_weights.view(num_layers, 1, 1, 1)
        
        
        # for every token i, we go through layers j, multiplyingsoftmax(s) 
        fused_embeddings = torch.sum(attn_weights * hidden_states, dim=0)

        return c * fused_embeddings  # Apply scaling factor



    def get_embedding(self,sentence):
        """ UPDATE ONCE DEBUGGING IS DONE!!!!!!!!!!!!!!!!"""
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence[0]) # 
            
        # return text as a list of tokens
        return [token.text for token in doc]




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