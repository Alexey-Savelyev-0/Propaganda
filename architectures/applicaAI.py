"""The ApplicaAI Span Identification architecture:
Take Roberta-CRF, train on gold data using vertribri loss.
Take the that model, label silver dataset.
Train new model on silver dataset using vertribri loss.
"""


""" Hyperparameters:
Dropout .1 
Attention dropout .1 
Max sequence length 256  
Batch size 8 
Learning rate 5e-4 
Number of steps 60k 
Momentum .9


"""




training_config = {
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "max_seq_length": 256,
    "batch_size": 8,
    "learning_rate": 5e-4,
    "max_steps": 20,
    "momentum": 0.9
}



import torch
import identification
from tqdm import tqdm
import numpy as np

NUM_ARTICLES = identification.NUM_ARTICLES

def train_applicaai_si(model, train_dataloader,silver_dataloader, steps=60000, save_model=True):
    # train on gold data
    progress_bar = tqdm(total=len(train_dataloader), desc="Training Progress")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["learning_rate"])
    global_step = 0
    model.train()
    for batch in train_dataloader:
        if global_step >= training_config["max_steps"]:
            break
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_labels, b_input_mask, b_ids = batch
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = output.get("loss")
        loss.backward()

        optimizer.step()
        model.zero_grad()
        global_step += 1
        progress_bar.update(1)
    progress_bar.close()
    
    # now using trained model, label silver data
    model.eval()

    tokenizer = identification.tokenizer
    token_predictions = []
    all_token_ids = []
    all_sentences = []
    silver_progress_bar = tqdm(total=len(silver_dataloader), desc="Labelling Silver Data")
    with torch.no_grad():
        for batch in silver_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_labels, b_input_mask, b_ids = batch

            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Extract token-wise logits
            logits = outputs.get("logits")  # Shape: (batch_size, seq_length, num_labels)
            preds = torch.argmax(logits, dim=-1)  # Get highest probability class for each token

            # Move to CPU for easy handling
            preds = preds.cpu().numpy()
            input_ids = b_input_ids.cpu().numpy()

            # Convert token IDs back to words (if tokenizer is available)
            sentences = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

            # Store results
            token_predictions.extend(preds)
            all_token_ids.extend(input_ids)
            all_sentences.extend(sentences)
            silver_progress_bar.update(1)
    silver_progress_bar.close()
    return token_predictions, all_token_ids, all_sentences

    
def run_train():
    TAGGING_SCHEME = identification.TAGGING_SCHEME
    BATCH_SIZE = identification.BATCH_SIZE
    num_labels = 2 + int(TAGGING_SCHEME =="BIO") + 2 * int(TAGGING_SCHEME == "BIOE")
    if identification.LANGUAGE_MODEL == "RoBERTa":
        from transformers import RobertaForTokenClassification
        model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=num_labels)
    elif identification.LANGUAGE_MODEL == "RoBERTa-CRF":
        from transformers import RobertaForTokenClassification
        # for now use roberta base
        model_base = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=num_labels)
        model = identification.RobertaCRF(model_base, num_labels)
    else:
        from transformers import BertForTokenClassification
        model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    gold_articles, article_ids = identification.read_articles('train-articles')
    gold_spans = identification.read_spans()
    gold_indices = np.arange(NUM_ARTICLES)

    silver_articles = identification.get_owt_articles()['text']
    silver_indices = np.arange(NUM_ARTICLES)
    silver_spans = [[] for _ in range(NUM_ARTICLES)]

    
    train_dataloader, train_sentences, train_bert_examples = identification.get_data(gold_articles, gold_spans, gold_indices)
    silver_dataloader, silver_sentences, silver_bert_examples = identification.get_data(silver_articles, silver_spans, silver_indices)
    token_predictions, all_token_ids, all_sentences = train_applicaai_si(model, train_dataloader, silver_dataloader, steps=60000, save_model=True)
    print(token_predictions)
    return model




run_train()