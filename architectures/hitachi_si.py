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
device = hitachi_utils.device
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
        if PLM == "RoBERTa":
            self.PLM = transformers.RobertaModel.from_pretrained('roberta-base',output_hidden_states=True)
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
        self.nlp = spacy.load("en_core_web_sm")
        # for auxillary task2 we need another similar model to HITACHI_SI to predict which sentences to train on

        self.CRF = CRF(3,batch_first=True)

    def forward(self,input_ids=None,lengths=None,token_type_ids = None, labels_TC = None, attention_mask = None):
        PLM_output,lengths,token_type_ids,labels_TC = self.get_sentence_inputs(input_ids,lengths = lengths,labels_TC = labels_TC,labels_span = token_type_ids)
        assert PLM_output.shape[0] == lengths.shape[0]
        
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        packed_input = pack_padded_sequence(PLM_output.float(), lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.BiLSTM(packed_input)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        BIO_output = self.FFN_BIO(lstm_output)
        padding_count = input_ids.shape[1] - BIO_output.shape[1]
        padding = (0, 0, 0, padding_count)
        BIO_output = F.pad(BIO_output, padding, "constant", 0)
        padding = (0, padding_count)
        token_type_ids= F.pad(token_type_ids, padding, "constant", -1)
        labels_TC = F.pad(labels_TC, padding, "constant", -1)
        if token_type_ids is not None:  # Training mode (return loss)
            
            mask = torch.tensor((token_type_ids != -1), dtype=torch.bool, device=BIO_output.device)
            assert (mask[:, 0] == 1).all(), "Error: Some sequences have mask[:, 0] == 0!"  # Create mask for ignoring padded tokens
            crf_loss = -self.CRF(BIO_output, token_type_ids, mask=mask, reduction='mean' )  # Compute CRF loss
            return crf_loss
        
        if labels_TC is not None and classification.TLC == True:  # Training mode (return loss)
            tc=self.FFN_TC(lstm_output)
        else:  
            predicted_labels = self.CRF.decode(BIO_output)  
            return predicted_labels
    def get_sentence_inputs(self,input_ids=None,lengths=None,labels_span = None,labels_TC = None):
        
        sentence_reps = []
        sentence_lengths = []
        sentence_span_labels = []
        sentence_tc_labels = []
        for i, sentence_ids in enumerate(input_ids):
            token_sentences= hitachi_utils.tokenizer.convert_ids_to_tokens(sentence_ids, skip_special_tokens=True)
            token_sentences = [token for token in token_sentences if token != hitachi_utils.tokenizer.pad_token or token != hitachi_utils.tokenizer.cls_token or token != hitachi_utils.tokenizer.sep_token]
            string_sentence = hitachi_utils.tokenizer.convert_tokens_to_string(token_sentences)
            string_sentence = string_sentence.replace("Ġ", " ")
            string_sentence = string_sentence.replace("ĊĊ", "\n")
            special_tokens = [
                    hitachi_utils.tokenizer.pad_token,
                    hitachi_utils.tokenizer.cls_token,
                    hitachi_utils.tokenizer.sep_token,
                    "<unk>",
                    "<s>"]
        
            for token in special_tokens:
                string_sentence = string_sentence.replace(token, "")
            sentence_rep, updated_labels, updated_tc_labels = self.get_token_representation(string_sentence,sentence_ids,labels_span[i],labels_TC[i])
            sentence_lengths.append(sentence_rep.shape[0])
            sentence_reps.append(sentence_rep)
            sentence_span_labels.append(updated_labels)
            sentence_tc_labels.append(updated_tc_labels)
        padded_reps = torch.nn.utils.rnn.pad_sequence(sentence_reps,batch_first=True)
        sentence_lengths = torch.tensor(sentence_lengths)
        sentence_span_labels = torch.nn.utils.rnn.pad_sequence(sentence_span_labels,batch_first=True)
        sentence_tc_labels = torch.nn.utils.rnn.pad_sequence(sentence_tc_labels,batch_first=True)
        return padded_reps, sentence_lengths, sentence_span_labels, sentence_tc_labels

       
    
    
    def get_token_representation(self,sentence,sentence_ids,labels_span,labels_tc):
        ### assume sentence is already tokenized
        """Token representation is obtained by concatting the plm representation, the PoS tag, and NE tag."""
        plm,labels,tc_labels = self.get_PLM_layer_attention(sentence,sentence_ids,labels_span=labels_span,labels_TC=labels_tc)
        
        ner, pos = self.get_special_tags(sentence)
        plm = plm.to(device)
        ner = ner.to(device)
        pos = pos.to(device)
        output = torch.cat((plm, ner, pos),dim=-1)
        # flatten out to 2d, as first dimension is always 1
        
        assert output.shape[0] == 1
        output = output.squeeze(0)
        return output,labels,tc_labels
    

    def get_PLM_layer_attention(self,sentence,sentence_ids,avg_subtokens=True,tokenizer=hitachi_utils.tokenizer,labels_span=None,labels_TC=None):
        """ To obtain input representations, we provide a layer-wise attention to fuse the outputs of PLM layers.
        To obtain the ith word, we sum PLM(i,j) over j, j being the layer index. In this sum
        we apply a softmax to a trainable parameter vector, which is multiplied by the output of the PLM layer.
        The concrete details may be found in the paper.
        """
        
        # for BERT
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
        offset = inputs["offset_mapping"].tolist()[0]
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        spacy_sentence = self.get_tokens(sentence)
        num_words = len(spacy_sentence) + 2
        token_to_word_mapping = {}
        # ensure token_to_word_mapping same format as before
        token_to_word_mapping[0] = 0
        first_token_indices = torch.full((num_words,), -1, dtype=torch.long, device=inputs["input_ids"].device)
        for token_idx, sub_offset in enumerate(offset):
            sub_start, sub_end = sub_offset
            mapped_word_idx = -1  # Default if no match
            
            # Find the spaCy word containing this subword's start position
            for word_idx, (_, word_start, word_end) in enumerate(spacy_sentence):
                if word_start <= sub_start < word_end:
                    mapped_word_idx = word_idx
                    break
    
            token_to_word_mapping[token_idx] = mapped_word_idx

        for token_idx, word_idx in token_to_word_mapping.items():
            if first_token_indices[word_idx] == -1:
                first_token_indices[word_idx] = token_idx
        
        # Handle [SEP] token's word (last word index)
        sep_word_idx = num_words -1 
        sep_token_idx = inputs["input_ids"].size(1) - 1  # Last token is [SEP]
        if first_token_indices[sep_word_idx] == -1:
            first_token_indices[sep_word_idx] = sep_token_idx
        
        # Fill remaining -1 (unmapped words) with 0 ([CLS])
        first_token_indices[first_token_indices == -1] = 0
        
        # Gather labels
        if labels_span is not None:
            word_labels_span = labels_span[first_token_indices]
        if labels_TC is not None:
            word_labels_TC = labels_TC[first_token_indices]



        device = next(self.PLM.parameters()).device
        inputs = {key: value.to(device) for key,value in inputs.items()}
            
        outputs = self.PLM(**inputs)
        hidden_states = outputs.hidden_states  # Tuple of shape (num_layers, batch_size, seq_len, hidden_dim)

        

        num_layers = len(hidden_states)
        batch_size, seq_len, hidden_dim = hidden_states[0].shape

        hidden_states = torch.stack(hidden_states, dim=0)

        if avg_subtokens == True:
            averaged_hidden_states = torch.zeros((num_layers, batch_size, num_words, hidden_dim), device=hidden_states.device)
            
            word_indices = torch.tensor(list(token_to_word_mapping.values()), device=hidden_states.device)
            assert word_indices.shape[0] == seq_len
            word_counts = torch.zeros((batch_size, num_words, 1), device=hidden_states.device)
            averaged_hidden_states = averaged_hidden_states.scatter_add(2, word_indices.view(1, 1, -1, 1).expand(num_layers, batch_size, -1, hidden_dim), hidden_states)
            ones = torch.ones((batch_size, seq_len, 1), device=hidden_states.device)
            word_counts = word_counts.scatter_add(1, word_indices.view(1, -1, 1).expand(batch_size, -1, 1), ones)
            word_counts[word_counts == 0] = 1 
            averaged_hidden_states /= word_counts.unsqueeze(0)
            hidden_states=averaged_hidden_states



        attn_weights = F.softmax(self.s, dim=-1) 
        attn_weights = attn_weights.view(num_layers, 1, 1, 1)
        
        
        # for every token i, we go through layers j, multiplyingsoftmax(s) 
        fused_embeddings = torch.sum(attn_weights * hidden_states, dim=0)
        # I don't know why this is required
        test = self.c.clone()
        output = torch.mul(fused_embeddings, test)
        return fused_embeddings,word_labels_span, word_labels_TC  # Apply scaling factor
    
    def get_special_tags(self,sentence):
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
    
    def get_tokens(self,sentence):
        
        doc = self.nlp(sentence) # 
            
        # return text as a list of tokens
        return [(token.text,token.idx, token.idx + len(token.text)) for token in doc]
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
    torch.autograd.set_detect_anomaly(True)
    hitachi_si = HITACHI_SI()

    if torch.cuda.is_available():
        hitachi_si.cuda()
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

    
    NUM_ARTICLES = hitachi_utils.NUM_ARTICLES
    articles = articles[0:NUM_ARTICLES]
    techniques = techniques[0:NUM_ARTICLES]
    spans = spans[0:NUM_ARTICLES]
    indices = np.arange(NUM_ARTICLES)
    eval_indices = indices[int(0.9 * NUM_ARTICLES):]
    train_indices = indices[:int(0.9 * NUM_ARTICLES)]

    dataloader, train_sentences, train_bert_examples = identification.get_data(articles, spans, train_indices,techniques=techniques)
    eval_dataloader, eval_sentences, eval_bert_examples = identification.get_data(articles, spans, eval_indices,techniques=techniques)

    #dataloader = identification.get_data(articles, spans, techniques, train_indices, PLM = hitachi_si.PLM, s = hitachi_si.s, c = hitachi_si.c)
    optimizer = optim.AdamW(hitachi_si.parameters(), lr=hitachi_utils.LEARNING_RATE, weight_decay=1e-2)
    total_loss = 0
    
    hitachi_si.train()
    for epoch_i in range(0, hitachi_utils.EPOCHS):
        total_loss,steps = (0,0)
        length = len(dataloader)
        for step, batch in enumerate(dataloader):
            #inputs, labels, masks, sentence_ids,techniques
            b_input_ids, b_labels,b_masks,b_ids, b_techniques = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)     
            b_techniques = b_techniques.to(device)   
            loss = hitachi_si(b_input_ids, 
                        token_type_ids = b_labels,
                        labels_TC = b_techniques)
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
    """
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
    """






if __name__ == "__main__":
    #model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
    hitachi_si_train()