import transformers
import torch
from torch import nn
import numpy as np
import spacy
# import crf
import torch.nn.functional as F
from torchcrf import CRF
from sklearn.model_selection import train_test_split
import classification

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

class FFN_TC(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, output_dim=15):
        super(FFN_TC, self).__init__()
        self.FFN = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) #output
            
        )
        crf = CRF(output_dim)
    def forward(self, x):
        nn_output = self.FFN(x)
        crf_output = self.crf(nn_output)
        return crf_output




class HITACHI_SI(nn.Module):
    def __init__(self,PLM="BERT", input_dim=None, hidden_dim=None, output_dim=None):
        super(HITACHI_SI, self).__init__()
        if PLM == "BERT":
            self.PLM = transformers.BertModel.from_pretrained('bert-base-uncased')

        self.BiLSTM = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.output_dim = output_dim
        self.linear = nn.Linear(hidden_dim*2, output_dim)
        self.relu = nn.ReLU()

        self.FFN_BIO = nn.Linear(input_dim, hidden_dim)
        self.FFN_TC = nn.Linear(input_dim, hidden_dim)
        # for auxillary task2 we need another similar model to HITACHI_SI to predict which sentences to train on

        self.CRF = CRF(output_dim)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,head_mask=None,inputs_embeds=None,labels=None,lengths=None):
        # before this is even ran, data should be preprocessed via get_token_representation
        lstm_output = self.BiLSTM(input_ids)
        technique_classification = self.FFN_TC(lstm_output)
        # how to get tc loss?
        bio_classification = self.FFN_BIO(lstm_output)

    
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

    def TLTC(self):
        """ Token Level technique classification"""
        pass

    def SLC(self):
        """Sentence Level Classification"""


    def get_embedding(self,sentence):
        """ UPDATE ONCE DEBUGGING IS DONE!!!!!!!!!!!!!!!!"""
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence[0]) # 
            
        # return text as a list of tokens
        return [token.text for token in doc]

    def get_special_tags(self,sentences):
        """ Tokenizes sentences, returns their special tags in dimensions (sentences, sentence-length, vector length)
        In Hitachi, special tags are: Part of Speech - PoS and Named Entities: One hot vector embeddings are generated for both
        If BERT is being used, 'special embeddings for CLS and SEP' are used.
        """


        nlp = spacy.load("en_core_web_sm")
        
        pos_tags = list(nlp.get_pipe("tagger").label_data)  
        ner_tags = list(nlp.get_pipe("ner").labels)  
        
        batch_pos_embeddings = []
        batch_ner_embeddings = []
        for sentence in sentences:
            doc = nlp(sentence)
            sentence_tokens = [token.text for token in doc]
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

    def get_token_representation(self,sentence,model,tokenizer):
        """Token representation is obtained by concatting the plm representation, the PoS tag, and NE tag."""
        plm = self.get_PLM_layer_attention(sentence,model,tokenizer)
        ner, pos = self.get_special_tags(sentence)

        return torch.cat((plm, ner, pos),dim=-1)


class FFN_SLC(HITACHI_SI):
    def __init__(self, input_dim, hidden_dim=200, output_dim=1):
        super(FFN_SLC, self).__init__()
        
    def forward(self, x):
        """
        sentence class = sigmoid(
        trainable vec transposed *
        FFN(BiLISTM(BoS) concat positional info)
        + bias
        )
        in theory bias+ trainable vec should be accounted for in a pytorch linear module
        """
        return self.BiLSTM(x)



def hitachi_si_train():

    articles, article_ids = classification.read_articles("train-articles")
    spans, techniques = classification.read_spans()
    NUM_ARTICLES = 10
    articles = articles[0:NUM_ARTICLES]
    spans = spans[0:NUM_ARTICLES]
    techniques = techniques[0:NUM_ARTICLES]
    indices = np.arange(NUM_ARTICLES)
    train_articles, eval_articles, train_spans, eval_spans, train_techniques, eval_techniques, train_indices, eval_indices = train_test_split(articles, spans, techniques, indices, test_size=0.2)
    train_dataloader = classification.get_data(train_articles, train_spans, train_techniques)
    eval_dataloader = classification.get_data(eval_articles, eval_spans, eval_techniques)
    classification.train(model, train_dataloader, eval_dataloader, epochs=2)




if __name__ == "__main__":
    model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    sentence = ["The quick brown fox jumps over the lazy dog."]
    sentence = ["##Tokenization is fascinating!"]
    hitachi_si = HITACHI_SI()
    #sentence = ["Can you pass me the salt?"]
    #sentence = "The slow."
    print(hitachi_si.get_token_representation(sentence,model, tokenizer))