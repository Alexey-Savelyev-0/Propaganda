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





import torch
import identification


def train_applicaai_si(model, train_dataloader, eval_dataloader, train_sentences, steps=60000, save_model=True):

    # Load data
    
    train_data = load_data("train")
    worse_training_data = load_data("test")

    # Train Roberta-CRF on gold data
    roberta_crf = RobertaCRF(vi)
    roberta_crf.train(train_data)

    # first combine the gold and silver data
    combined_data = train_data + worse_training_data


    # Label silver data
    silver_data = roberta_crf.label(combined_data)

    # Train Roberta-CRF on silver data
    # what is converged?
    step_count = 0
    while step_count < steps:
        roberta_crf.train(silver_data)
