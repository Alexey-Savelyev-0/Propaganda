import transformers
import torch
from torch import nn
import numpy as np
import datetime
import os
import spacy
from tqdm import tqdm, trange

from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torchcrf import CRF
from sklearn.model_selection import train_test_split
import identification as identification
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import classification as classification
import identification as identification
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold

device = identification.device


class HITACHI_SI(nn.Module):
    def __init__(self,PLM=identification.LANGUAGE_MODEL, input_dim= 1099, hidden_dim=600, output_dim=100,create_slc=True,weight_pos=torch.tensor([1], dtype=torch.float)):
        super(HITACHI_SI, self).__init__()
        if PLM == "RoBERTa":
            self.PLM = transformers.RobertaModel.from_pretrained('roberta-large',output_hidden_states=True)
            self.PLM.add_pooling_layer = False
            # +1 means we currently include embedding layer in the attnetion weights - not sure that is correct
        else:
            self.PLM = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
            self.PLM.to(device)
        # initially we freeze the LLM to stabilise the model
        for param in self.PLM.parameters():
            param.requires_grad = True
        num_layers = self.PLM.config.num_hidden_layers +1
        self.s_si = nn.Parameter(torch.randn(num_layers))  # Attention weights for both span identification and sentence level classification
        self.s_slc = nn.Parameter(torch.rand(num_layers))

            # c is a scalar, s is a vector of dim num_layers 
        self.c_si = nn.Parameter(torch.ones(1)) 
        self.c_slc = nn.Parameter(torch.ones(1)) 

        self.BiLSTM = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True,batch_first=True)
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.FFN_BIO = nn.Sequential(
        nn.Linear(hidden_dim * 2, 200),  
        nn.ReLU(),
        nn.Linear(200, 3)   
        )
        self.FFN_TC = nn.Sequential(
        nn.Linear(hidden_dim * 2, 200),   
        nn.ReLU(),
        nn.Linear(200, 15)                
    )
        if create_slc:
            self.SLC = SLC(weight_pos=weight_pos)
        # for auxillary task2 we need another similar model to HITACHI_SI to predict which sentences to train on

        self.CRF_BIO = CRF(3,batch_first=True)
        self.CRF_TC = CRF(15,batch_first=True)
        self.to(device)    
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self,input_ids=None,input_lengths=None,token_type_ids = None, labels_TC = None, token_enrichment=None,token_mapping=None, attention_mask = None):
        PLM_output_SI, PLM_output_SLC=self.get_token_representation(input_ids,context=token_enrichment,mapping=token_mapping,mask=attention_mask)
        lengths = attention_mask.sum(dim=1)
        if identification.SLC:
            if token_type_ids is not None:
                slc_output, slc_loss = self.SLC(PLM_output_SLC,lengths=lengths,token_type_ids=token_type_ids)
            else:
                slc_output, _ = self.SLC(PLM_output_SLC,lengths = lengths)
        
        #packed_input = pack_padded_sequence(PLM_output_SI.float(), lengths.cpu(), batch_first=True, enforce_sorted=False)
        #unsorted_indices = packed_input.unsorted_indices
        lstm_output, _ = self.BiLSTM(PLM_output_SI)
        #lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        #lstm_output = lstm_output[unsorted_indices]
        BIO_output = self.FFN_BIO(lstm_output)
        tc=self.FFN_TC(lstm_output)
        padding_count = input_ids.shape[1] - BIO_output.shape[1]
        
        padding = (0, 0, 0, padding_count)
        #BIO_output = F.pad(BIO_output, padding, "constant", 0)
        #tc = F.pad(tc, padding, "constant", 0)
        padding = (0, padding_count)
        mask = attention_mask > 0
        if token_type_ids is not None:  # Training mode (return loss)
            tags_safe = token_type_ids.clone()
            tags_safe[tags_safe == -1]   = 0
            tags_safe[tags_safe == -100] = 0
            #assert (attention_mask[:, 0] == 1).all()
            if identification.CRF:
                bio_loss = -self.CRF_BIO.forward(BIO_output, tags_safe, mask=mask, reduction='none' )  # Compute CRF loss
            else:
                #print(BIO_output.shape)
                #print(token_type_ids.shape)
                BIO_output = BIO_output.permute(0, 2, 1)
                bio_loss = self.criterion(BIO_output, token_type_ids)
                # assign classes to logits of FFN then calc cross entropy loss
            if labels_TC is not None and identification.TLC == True:  # Training mode (return loss)
            # not clear W and b need to be added, although I suppose it can't hurt
                if identification.CRF:
                    tc_loss = -self.CRF_TC.forward(tc,labels_TC,mask=mask, reduction='none')
                    # assign classes to logits of FFN then calc cross entropy loss
                else:
                    tc = tc.permute(0, 2, 1)
                    tc_loss = self.criterion(tc, labels_TC)
        else: 
            # slc_output is of dim (b) and contains 1s and 0s. we only want to return the
            # BIO predicted labels if that batch has a 1.
            if identification.CRF:
                predicted_labels = self.CRF_BIO.decode(BIO_output,mask=mask)  
            else:
                predicted_labels = torch.argmax(BIO_output, dim=-1) 
            final_labels = []
            for keep_flag, labels in zip(slc_output.tolist(), predicted_labels):
                if keep_flag == 1:
                #if True:
                    final_labels.append(labels)
                    
                else:
                    final_labels.append([0]*256)
            return final_labels
        if identification.SLC:
            has_span = ((token_type_ids == 1) | (token_type_ids == 2)).any(dim=1) 
            has_span_float = has_span.float()  
            span_weights = has_span_float               # shape (B,)

            gated_bio = span_weights * bio_loss         # only keep if has_span=1
            gated_tc  = span_weights * tc_loss 
            #return bio_loss.mean()
            return (slc_loss +  gated_bio ).mean()


    
    
    def get_token_representation(self,sentence_ids,context=None,mapping = None,mask = None):
        """
        Inputs: batch of setences, batch of spans, batch of techniques, batch of cotnext tensors, mapping
        Token representation is obtained by concatting the plm representation, the PoS tag, and NE tag."""
        plm_si, plm_slc = self.get_PLM_layer_attention(sentence_ids,mapping=mapping, mask = mask)
        
        plm_si = plm_si.to(device)
        plm_slc = plm_slc.to(device)
        concatted_si = torch.cat((plm_si, context),dim=-1)
        concatted_slc = torch.cat((plm_slc, context),dim=-1)
        # flatten out to 2d, as first dimension is always 1
        
        return concatted_si,concatted_slc
    

    def get_PLM_layer_attention(self,sentence_ids,avg_subtokens=False,mapping=None, mask=None):
        """ To obtain input representations, we provide a layer-wise attention to fuse the outputs of PLM layers.
        To obtain the ith word, we sum PLM(i,j) over j, j being the layer index. In this sum
        we apply a softmax to a trainable parameter vector, which is multiplied by the output of the PLM layer.
        One output is for the main task e.g. Span Identification, and One is for sentence Level Classification, which uses a seperate attention mechanism.
        """
        
        outputs = self.PLM(
    input_ids=sentence_ids,
    attention_mask=mask, output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states  # Tuple of shape (num_layers, batch_size, seq_len, hidden_dim)

        

        num_layers = len(hidden_states)
        batch_size, seq_len, hidden_dim = hidden_states[0].shape

        hidden_states = torch.stack(hidden_states, dim=0)

        output_si =  outputs.last_hidden_state
        attn_weights_si = F.softmax(self.s_si, dim=-1) 
        attn_weights_si = attn_weights_si.view(num_layers, 1, 1, 1)
        attn_weights_slc = F.softmax(self.s_slc, dim=-1) 
        attn_weights_slc = attn_weights_slc.view(num_layers, 1, 1, 1)
        
        
        fused_embeddings_si = torch.sum(attn_weights_si * hidden_states, dim=0)
        fused_embeddings_slc = torch.sum(attn_weights_slc * hidden_states, dim=0)
        output_si = torch.mul(fused_embeddings_si, self.c_si)
        output_slc = torch.mul(fused_embeddings_slc,self.c_slc)
        return output_si,output_slc
    

    


    def get_special_tags(self,sentences):
        """ Tokenizes sentences, returns their special tags in dimensions (sentences, sentence-length, vector length)
        In Hitachi, special tags are: Part of Speech - PoS and Named Entities: One hot vector embeddings are generated for both
        If BERT is being used, 'special embeddings for CLS and SEP' are used.
        """

        
        
        pos_tags = self.pos_labels
        ner_tags = self.ner_labels
        doc = self.nlp(sentences)
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
            pos_one_hot = torch.zeros(len(pos_tags)+2)
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
    
    def get_tokens(self,sentence):
        
        doc = self.nlp(sentence) # 

        return [(token.text,token.idx, token.idx + len(token.text)) for token in doc]





class SLC(nn.Module):
    def __init__(
        self,
          # frozen encoder supplied by the parent
        #input_dim: int = 843,
        # +75
        input_dim = 1099,
        hidden_dim: int = 600,
        threshold: float = 0.5,
        weight_pos=None                 
    ):
        super(SLC, self).__init__()
        # encoder
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers = 2,
            bidirectional=True,
            batch_first=True
        )

        self.classifier = nn.Sequential(
        nn.Linear(hidden_dim * 2, 500),  
        nn.ReLU(),
        nn.Linear(500, 1)   
        )
        # loss
        self.loss_fn = nn.BCEWithLogitsLoss(reduction = 'none')
        self.threshold = threshold

    def forward(
        self,
        plm_output: torch.Tensor,                # (B, T, input_dim)
        lengths:     torch.Tensor,               # (B,)  – sentence lengths
        token_type_ids: torch.Tensor = None,     # (B, T)        # (B,)  – 0 or 1, optional
    ):
        labels = None
        
        #packed_input = pack_padded_sequence(
            #plm_output, lengths.cpu(),
            #batch_first=True, enforce_sorted=False
        #)
        #unsorted_indices = packed_input.unsorted_indices

        packed_out, _ = self.bilstm( plm_output)
        #lstm_out, _   = pad_packed_sequence(packed_out, batch_first=True)
        #lstm_out = lstm_out[unsorted_indices]
        cls_repr = packed_out[:, 0, :]                  
        logits   = self.classifier(cls_repr).squeeze(-1) 

        if token_type_ids is not None:
            labels = ((token_type_ids == 1) | (token_type_ids == 2)).any(dim=1)     

        if labels is not None:                            
            loss  = self.loss_fn(logits, labels.float())

            preds = (torch.sigmoid(logits) > self.threshold).long()
            return preds,loss

        else:
                                      # INFERENCE
            probs = torch.sigmoid(logits)                
            preds = (probs > self.threshold).long()       
            return preds, None




def hitachi_si_train():
    torch.cuda.empty_cache()

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
        
    techniques = identification.get_si_techniques(spans,tc_spans_techniques)

    import sys
    NUM_ARTICLES = identification.NUM_ARTICLES
    articles = articles[0:NUM_ARTICLES]
    article_ids = article_ids[0:NUM_ARTICLES]
    techniques = techniques[0:NUM_ARTICLES]
    spans = spans[0:NUM_ARTICLES]
    indices = np.arange(NUM_ARTICLES)
    train_indices = indices[:int(0.9 * NUM_ARTICLES)]
    eval_indices = indices[int(0.9 * NUM_ARTICLES):]



    #_,dataset, train_sentences, train_bert_examples = identification.get_data_bert(articles, spans, indices,techniques=techniques,return_dataset=True)

   
    total_loss = 0
    splits = 5
    print("begin training")
    

    train_dataloader, train_sentences, train_bert_examples = identification.get_data_bert(articles, spans, train_indices,techniques=techniques, shuffle=True)
    eval_dataloader, eval_sentences, eval_bert_examples = identification.get_data_bert(articles, spans, eval_indices,techniques=techniques, shuffle=False)
    n_pos = 0
    n_neg = 0
    for sentence in train_sentences:
            if 1 in sentence.labels or 2 in sentence.labels:
                n_pos+=1
            else:
                n_neg+=1
            
    WEIGHT_POS = n_neg /n_pos       # heavier weight for positives 
    WEIGHT_POS 
    WEIGHT_POS = torch.tensor([WEIGHT_POS], dtype=torch.float)

    hitachi_si = HITACHI_SI(weight_pos=WEIGHT_POS)
    if torch.cuda.is_available():
            hitachi_si.cuda()
    hitachi_si.to(device)
    optimizer = optim.AdamW([
    { "params": hitachi_si.PLM.parameters(),    "lr": 2.9e-6 },
    { "params": hitachi_si.BiLSTM.parameters(), "lr": 6e-4 },
    { "params": hitachi_si.CRF_BIO.parameters(), "lr": 6e-4 },
    { "params": hitachi_si.CRF_TC.parameters(), "lr": 6e-4 },
    { "params": hitachi_si.FFN_BIO.parameters(), "lr": 6e-4 },
    { "params": hitachi_si.FFN_TC.parameters(), "lr": 6e-4 },
    {"params": hitachi_si.SLC.parameters(), "lr": 6e-4 },
    { "params": hitachi_si.c_si, "lr": 6e-4 },
    { "params": hitachi_si.c_slc, "lr": 6e-4 },
    { "params": hitachi_si.s_si, "lr": 6e-4 },
    { "params": hitachi_si.s_slc, "lr": 6e-4 },
], 
                                weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
              
    for epoch_i in trange(0, identification.EPOCHS):
            if epoch_i % 2 == 0:
                for param in hitachi_si.PLM.parameters():
                    param.requires_grad = True
            print("new epoch")
            hitachi_si.train()
            total_loss,steps = (0,0)
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                #inputs, labels, masks, sentence_ids,techniques
                b_input_ids,b_labels,b_context,b_masks,b_ids, b_techniques, b_mapping = batch
                b_input_ids = b_input_ids.to(device)
                b_masks = b_masks.to(device)
                b_labels = b_labels.to(device)     
                b_techniques = b_techniques.to(device)   
                b_context = b_context.to(device)
                b_mapping = b_mapping.to(device)
                loss = hitachi_si(b_input_ids, 
                            token_type_ids = b_labels,
                            labels_TC = b_techniques,
                            token_enrichment = b_context,
                            token_mapping = b_mapping,
                            attention_mask = b_masks)
                total_loss += loss.detach().item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hitachi_si.parameters(), max_norm=5.0)  # Gradient clipping
                optimizer.step()
                steps+=1
            for name, param in hitachi_si.named_parameters():
                if param.grad is None:
                    print(f"No grad for {name}")
                elif torch.all(param.grad == 0):
                    print(f"Zero grad for {name}")
            print(f"Epoch{epoch_i}: Avg Loss{total_loss/steps}")
            hitachi_si.eval()
            """
            identification.get_score(hitachi_si,
                train_dataloader,
                train_sentences,
                train_bert_examples,
                mode="eval",
                article_ids=article_ids,
                indices=train_indices,model_type = 'n/a')
            """
            identification.get_score(hitachi_si,
            eval_dataloader,
            eval_sentences,
            eval_bert_examples,
            mode="eval",
            article_ids=article_ids,
            indices=eval_indices,model_type = 'n/a')
            scheduler.step()
            if identification.SAVE_MODEL:
                    model_name = 'hitachi_si_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.pt'
                    torch.save(hitachi_si, os.path.join(identification.model_dir, model_name))
                    print("Model saved:", model_name)
    






if __name__ == "__main__":
    hitachi_si_train()