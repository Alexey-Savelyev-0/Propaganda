import transformers
import torch
from torch import nn
import numpy as np
import datetime
import os
import spacy
# import crf
import torch.nn.functional as F
from torchcrf import CRF
from sklearn.model_selection import train_test_split
import identification.hitachi_utils as hitachi_utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import classification.hitachi_utils as classification
import identification
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

class SLC(nn.Module):
    def __init__(self,PLM=hitachi_utils.LANGUAGE_MODEL, input_dim=840, hidden_dim=600, output_dim=15):
        super(SLC, self).__init__()
        self.FFN = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) #output
            
        )
        self.BiLSTM = nn.LSTM(input_dim, hidden_dim, bidirectional=True,batch_first=True)
        crf = CRF(output_dim)
    def forward(self, input_ids, lengths, pos_features, token_type_ids=None):
        packed_input = pack_padded_sequence(input_ids, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.BiLSTM(packed_input)
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)
        
        # Extract BoS token hidden state (first token)
        bos_output = lstm_output[:, 0, :]  # Shape: (batch_size, hidden_dim * 2)
        
        # Concatenate structural features (e.g., positional & length info)
        bos_output = torch.cat((bos_output, pos_features), dim=1)
        
        # Pass through FFN
        ff_output = self.FFN(bos_output)  # Shape: (batch_size, hidden_dim)
        
        # Compute the final classification output using v and b
        cls_output = torch.sigmoid(self.v(ff_output))  # Shape: (batch_size, 1)
        
        # CRF layer for sequence labeling
        bio_logits = self.crf(lstm_output)
        
        return cls_output, bio_logits




class HITACHI_SI(nn.Module):
    def __init__(self,PLM=hitachi_utils.LANGUAGE_MODEL, input_dim=840, hidden_dim=600, output_dim=15):
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
        self.SLC = SLC()
        # for auxillary task2 we need another similar model to HITACHI_SI to predict which sentences to train on

        self.CRF = CRF(3,batch_first=True)

    def forward(self,input_ids=None,lengths=None,token_type_ids = None, labels_TC = None, attention_mask = None):
        # before this is even ran, data should be preprocessed via get_token_representation
        # input of shape (batch_size, max_seq_len, input_dim)
        # labels is list of tensors of BIO tags
        if lengths is None:
            lengths = (input_ids != 0).sum(dim=1).long()
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        packed_input = pack_padded_sequence(input_ids.float(), lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.BiLSTM(packed_input)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        BIO_output = self.FFN_BIO(lstm_output)
        if token_type_ids is not None:  # Training mode (return loss)
            if isinstance(token_type_ids, list):  # Convert list to tensor
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=BIO_output.device)

            mask = torch.tensor((token_type_ids != -1), dtype=torch.bool, device=BIO_output.device)
            assert (mask[:, 0] == 1).all(), "Error: Some sequences have mask[:, 0] == 0!"  # Create mask for ignoring padded tokens
            crf_loss = -self.CRF(BIO_output, token_type_ids, mask=mask, reduction='mean' )  # Compute CRF loss
            return crf_loss
        
        if labels_TC is not None and classification.TLC == True:  # Training mode (return loss)
            tc=self.FFN_TC(lstm_output)
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



    

def hitachi_si_train():
    torch.autograd.set_detect_anomaly(True)
    hitachi_si = HITACHI_SI()

    if torch.cuda.is_available():
        hitachi_si.cuda()
        #inputs = inputs.cuda() 
        #target = target.cuda()

    articles, article_ids = identification.read_articles("train-articles")
    """ note that the spans outlined in the classification section are 
    similar but different-> additional ones are added in the classification section.
    Therefore only use spans/techniques if span exists in identification section"""
    
    spans = identification.read_spans()
    tc_spans, techniques = classification.read_spans()
    tc_spans_techniques = {}
    for i in range(len(tc_spans)):
        for j in range(len(tc_spans[i])):
            tc_spans_techniques[(i,tc_spans[i][j])] = techniques[i][j]
        
    techniques = get_si_techniques(spans,tc_spans_techniques)

    
    NUM_ARTICLES = hitachi_utils.NUM_ARTICLES
    articles = articles[0:NUM_ARTICLES]
    techniques = techniques[0:NUM_ARTICLES]
    spans = spans[0:NUM_ARTICLES]
    indices = np.arange(NUM_ARTICLES)
    eval_indices = indices[int(0.9 * NUM_ARTICLES):]
    train_indices = indices[:int(0.9 * NUM_ARTICLES)]

    train_dataloader, train_sentences, train_bert_examples = identification.get_data(articles, spans, train_indices)
    eval_dataloader, eval_sentences, eval_bert_examples = identification.get_data(articles, spans, eval_indices)

    dataloader = identification.get_data_hitachi(articles, spans, techniques, train_indices, PLM = hitachi_si.PLM, s = hitachi_si.s, c = hitachi_si.c)
    optimizer = optim.AdamW(hitachi_si.parameters(), lr=hitachi_utils.LEARNING_RATE, weight_decay=1e-2)
    total_loss = 0
    
    hitachi_si.train()
    for epoch_i in range(0, hitachi_utils.EPOCHS):
        total_loss,steps = (0,0)
        length = len(dataloader)
        for step, batch in enumerate(dataloader):
            
            b_input_ids, b_lengths, b_labels, b_techniques = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)        
            loss = hitachi_si(b_input_ids, 
                        token_type_ids = b_labels,
                        lengths=b_lengths)
            total_loss += loss.detach().item()
            # not a fan, no idea why retain_graph needs to be true but we ball
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(hitachi_si.parameters(), max_norm=5.0)  # Gradient clipping
            optimizer.step()
            optimizer.zero_grad()
            hitachi_si.zero_grad(
            )
            steps+=1
        print(f"Epoch{epoch_i}: Avg Loss{total_loss/steps}")

    identification.get_score(hitachi_si,
        eval_dataloader,
        eval_sentences,
        eval_bert_examples,
        mode="eval",
        article_ids=article_ids,
        indices=eval_indices)
    if identification.SAVE_MODEL:
      model_name = 'hitachi_si_' + str(datetime.datetime.now()) + '.pt'
      torch.save(hitachi_utils.model, os.path.join(identification.model_dir, model_name))
      print("Model saved:", model_name)







if __name__ == "__main__":
    #model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
    hitachi_si_train()