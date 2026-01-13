import argparse
import transformers
from tqdm import trange
from transformers import BertForTokenClassification
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from LSTM import LSTMClassifier
import re
import identification
import classification
import datetime
from collections import Counter
import re
import os
from collections import Counter



def tokenize(tokens):
    return [re.sub(r"[^a-z0-9]", "", t.lower()) or "<unk>" for t in tokens]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
PAD_LABEL = -100
def main(args):
    if args.test:
        raw = load_dataset("conll2003")
        train_split = raw["train"]
        valid_split = raw["validation"]
        test_split  = raw["test"]
        label_names = raw["train"].features["ner_tags"].feature.names
        n_labels    = len(label_names)
        counter = Counter(tok for ex in train_split for tok in tokenize(ex["tokens"]))
        VOCAB_SIZE = 30_000
        most_common = counter.most_common(VOCAB_SIZE - 2)
        word2idx = {w: i+2 for i, (w, _) in enumerate(most_common)}
        word2idx["<pad>"] = 0
        word2idx["<unk>"] = 1
        pad_idx = word2idx["<pad>"]
        unk_idx = word2idx["<unk>"]
        pad_label_idx = -100

        MAX_LEN = 128
        class NERDataset(Dataset):
            def __init__(self, split):
                self.data = split
            def __len__(self):
                return len(self.data)
            def __getitem__(self, i):
                toks  = tokenize(self.data[i]["tokens"])[:MAX_LEN]
                tags  = self.data[i]["ner_tags"][:MAX_LEN]
                length = len(toks)
                ids = [word2idx.get(t, unk_idx) for t in toks] + [pad_idx]*(MAX_LEN-length)
                tags = tags + [pad_label_idx]*(MAX_LEN-length)
                return torch.tensor(ids), torch.tensor(tags), length

        def collate_fn(batch):
            ids, tags, lengths = zip(*batch)
            return torch.stack(ids), torch.stack(tags), torch.tensor(lengths)
        train_loader = DataLoader(NERDataset(train_split), batch_size=32, shuffle=True,  collate_fn=collate_fn)
        valid_loader = DataLoader(NERDataset(valid_split), batch_size=32, shuffle=False, collate_fn=collate_fn)
        test_loader  = DataLoader(NERDataset(test_split),  batch_size=32, shuffle=False, collate_fn=collate_fn)

    else:
        train_loader,valid_loader = prep_data()
        n_labels=3


    if args.lstm:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMClassifier(
            hidden_dim=256,
            n_layers=2,
            bidirectional=True,
            dropout=0.3,
            output_dim=n_labels,
        )
        model.to(device)
        

        best_valid = float("inf")
        time= str(datetime.datetime.now())
        model_name = f"si_lstm{time}.pt"
        for epoch in trange(20, desc="Epoch"):
            tr_loss, tr_acc = run_epoch(train_loader, model,train=True)
            vl_loss, vl_acc = run_epoch(valid_loader,model, train=False)
            if vl_loss < best_valid:
                best_valid = vl_loss
                
                torch.save(model.state_dict(), os.path.join(identification.model_dir, model_name))
            print(f"Epoch {epoch}: Train loss={tr_loss:.4f}, acc={tr_acc:.4f} | Valid loss={vl_loss:.4f}, acc={vl_acc:.4f}")

        # === Evaluation ===
        model.load_state_dict(torch.load(model_name))
        te_loss, te_acc = run_epoch(test_loader, model,train=False)
        print(f"Test loss={te_loss:.4f}, token-level accuracy={te_acc:.4f}")
    

def run_epoch(loader,model, train=True):
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            if train:
                model.train()
            else:
                model.eval()
            total_loss, total_corr, total_tokens = 0, 0, 0
            for step, batch in enumerate(loader):
                b_input_ids, b_labels,b_masks,b_ids, b_techniques = batch
                b_input_ids = b_input_ids.to(device)
                b_labels = b_labels.to(device)
                b_masks = b_masks.to(device)     
                b_techniques = b_techniques.to(device)
                 
                n_labels = 3
                logits = model(b_input_ids, b_masks,padding_length=MAX_LEN)  # [B, L, n_labels]
                loss = criterion(logits.view(-1, n_labels), b_labels.view(-1))
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                lengths = b_masks.sum(dim=1)
                total_loss += loss.item() * lengths.sum().item()
                preds = logits.argmax(-1)
                """
                print(f"og input ids: {b_input_ids}")
                print(f"mask: {b_masks}")
                print(f"og labels: {b_labels}")
                print(f"og pred: {preds}")
                """
                
                total_corr  += ((preds == b_labels) & b_masks).sum().item()
                total_tokens+= b_masks.sum().item()
            return total_loss/total_tokens, total_corr/total_tokens



def prep_data():
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
    
    NUM_ARTICLES = 370
    
    articles = articles[0:NUM_ARTICLES]
    techniques = techniques[0:NUM_ARTICLES]
    spans = spans[0:NUM_ARTICLES]
    indices = np.arange(NUM_ARTICLES)
    eval_indices = indices[int(NUM_ARTICLES*0.8):]
    train_indices = indices[:int(0.8 * NUM_ARTICLES)]
    dataloader, train_sentences, train_bert_examples = identification.get_data_bert(articles, spans, train_indices,techniques=techniques)
    eval_dataloader, eval_sentences, eval_bert_examples = identification.get_data_bert(articles, spans, eval_indices,techniques=techniques)
    return dataloader,eval_dataloader



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My awesome tool")
    parser.add_argument("-o", "--output", default="out.txt",
                        help="output file (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable verbose logging")
    parser.add_argument("-l", "--lstm",action="store_true", help="enables training of the LSTM defined in LSTM.py")
    parser.add_argument("-t","--test",action="store_true", help="run training on test data")
    args = parser.parse_args()
    print(f"Reading from nothing, writing to {args.output}")
    main(args)