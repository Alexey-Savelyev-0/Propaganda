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
    "num_steps": 60000,
    "momentum": 0.9
}



import torch
import identification
import tqdm
import numpy as np


def train_applicaai_si(model, train_dataloader, eval_dataloader, train_sentences, steps=60000, save_model=True):
    # train on gold data
    progress_bar = tqdm(total=training_config["max_steps"], desc="Training Progress")
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

    
    articles = get_owt_articles()['text']

    indices = np.arange(NUM_ARTICLES)
    print(articles[1])
    spans = [[] for _ in range(NUM_ARTICLES)]
    get_data(articles,spans, indices)
    # now using trained model, label silver data




def label_silver_data(model, silver_dataloader):
    pass
        
