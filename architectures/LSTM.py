# UNUSED IN FINAL REPORT - developed for further discussion but ran out of time to properly train/discuss. Feel free to ignore this.


import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
    
        for p in self.bert.parameters():
            p.requires_grad = False
       
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,   # 768
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,  # PyTorch rule
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )

    def forward(
        self,
        input_ids: torch.Tensor,       
        attention_mask: torch.Tensor ,  
        padding_length: int = 256,        
        token_type_ids = None   
    ):
        # ---- BERT ----
        with torch.no_grad():          
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        embedded = bert_out.last_hidden_state          

        # ---- pack/pad for LSTM ----
        seq_lens = attention_mask.sum(dim=1)           
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=padding_length
        )

        logits = self.fc(self.dropout(out))            # [B, L, output_dim]
        return logits