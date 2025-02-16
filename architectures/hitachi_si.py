import transformers
import torch
from torch import nn
import numpy as np
import spacy
# import crf
import torch.nn.functional as F
from torchcrf import CRF
from sklearn.model_selection import train_test_split
import identification.hitachi_utils as identification
from torch.nn.utils.rnn import pack_padded_sequence
from torch import optim
device = identification.device
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
    def __init__(self,PLM="BERT", input_dim=840, hidden_dim=600, output_dim=15):
        super(HITACHI_SI, self).__init__()
        if PLM == "BERT":
            self.PLM = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
            # +1 means we currently include embedding layer in the attnetion weights - not sure that is correct
            num_layers = self.PLM.config.num_hidden_layers +1
            self.s = nn.Parameter(torch.randn(num_layers))  # Attention weights
            # c is a scalar, s is a vector of dim num_layers 
            self.c = nn.Parameter(torch.ones(1)) 

        self.BiLSTM = nn.LSTM(input_dim, hidden_dim, bidirectional=True,batch_first=True)
        self.output_dim = output_dim
        self.relu = nn.ReLU()

        self.FFN_BIO = nn.Linear(hidden_dim*2, 3)
        self.FFN_TC = nn.Linear(input_dim, hidden_dim)
        # for auxillary task2 we need another similar model to HITACHI_SI to predict which sentences to train on

        self.CRF = CRF(3,batch_first=True)

    def forward(self,input_ids=None,lengths=None,labels_BIO = None, labels_TC = None):
        # before this is even ran, data should be preprocessed via get_token_representation
        # input of shape (batch_size, max_seq_len, input_dim)
        # labels is list of tensors of BIO tags
        print("input_ids",input_ids.shape)
        print("lengths",lengths.shape)
        print("labels_BIO",labels_BIO.shape)
        packed_input = pack_padded_sequence(input_ids, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.BiLSTM(packed_input)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        BIO_output = self.FFN_BIO(lstm_output)
        if labels_BIO is not None:  # Training mode (return loss)
            if isinstance(labels_BIO, list):  # Convert list to tensor
                labels_BIO = torch.tensor(labels_BIO, dtype=torch.long, device=BIO_output.device)

            mask = torch.tensor((labels_BIO != -1), dtype=torch.bool, device=BIO_output.device)
            assert (mask[:, 0] == 1).all(), "Error: Some sequences have mask[:, 0] == 0!"  # Create mask for ignoring padded tokens
            print("mask",mask)  
            crf_loss = -self.CRF(BIO_output, labels_BIO, mask=mask, reduction='mean' )  # Compute CRF loss
            return crf_loss
        else:  
            predicted_labels = self.CRF.decode(BIO_output)  
            return predicted_labels

    
    # test that this behaves in the way you expect it to.
    # at least dimensions are what they should be
    def TLTC(self):
        """ Token Level technique classification"""
        pass

    def SLC(self):
        """Sentence Level Classification"""
        pass


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
    hitachi_si = HITACHI_SI()
    articles, article_ids = identification.read_articles("train-articles")    
    spans = identification.read_spans()
    articles = articles[0:identification.NUM_ARTICLES]
    spans = spans[0:identification.NUM_ARTICLES]
    indices = np.arange(identification.NUM_ARTICLES)
    train_indices = indices[:int(0.9 * identification.NUM_ARTICLES)]
    dataloader = identification.get_data_hitachi(articles, spans, train_indices, PLM = hitachi_si.PLM, s = hitachi_si.s, c = hitachi_si.c)
    optimizer = optim.AdamW(hitachi_si.parameters(), lr=identification.LEARNING_RATE, weight_decay=1e-2)
    epochs = 1
    total_loss = 0
    for epoch_i in range(0, epochs):
        hitachi_si.train()
        for step, batch in enumerate(dataloader):
            b_input_ids, b_lengths, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)        
            loss = hitachi_si(b_input_ids, 
                        labels_BIO = b_labels,
                        lengths=b_lengths)
            loss.requires_grad = True
            print(f"Step {step} loss: {loss.item()}")
            total_loss += loss.item()

            loss.backward()
            print(f"Step {step} loss: {loss.item()}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
            optimizer.step()





if __name__ == "__main__":
    model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
    
    #sentence = ["The quick brown fox jumps over the lazy dog."]
    #sentence = ["##Tokenization is fascinating!"]

    hitachi_si_train()