from torchcrf import CRF
import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', lower_case=True)
""" supposedly already inherently used vetebri loss - confirm with own research."""


class RobertaCRF(nn.Module):
    def __init__(self, bert,num_labels, **kwargs):
        super(RobertaCRF, self).__init__()
        self.roberta = bert
        dropout = kwargs.get("dropout", 0.1)
        attention_dropout = kwargs.get("attention_dropout", 0.1)
        



        # Linear layer for converting RoBERTa output to label space
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_labels)

        # CRF layer for sequence labeling
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids,token_type_ids, attention_mask, labels=None):
        # Pass input through RoBERTa model
        # doesn't the outputs variable already contain the last hidden state?
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply linear layer to get scores for each label
        emissions = self.fc(sequence_output)
        attention_mask = attention_mask.bool()
        if labels is not None:
            # Ensure labels match the sequence length of emissions
            labels = labels[:, :emissions.size(1)]
            
            # Compute the negative log likelihood during training
            loss = -self.crf(emissions, labels, mask=attention_mask, reduction='mean')
            return {
                "loss": loss,
                "logits": emissions,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
            }
        else:
            # Decode the best label sequence during inference
            predictions = self.crf.decode(emissions, mask=attention_mask)
            return {
                "logits": emissions,
                "predictions": predictions,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
            }

        

if __name__ == "__main__":
    model_name = "roberta-base"  # You can use other RoBERTa variants
    num_labels = 9  # Adjust based on your dataset
    model_base = RobertaModel.from_pretrained(model_name)
    model = RobertaCRF(model_base, num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)


    config = {
        "dropout": 0.2,
        "attention_dropout": 0.2,
        "learning_rate": 5e-5,
        "batch_size": 16,
    }

    # Example input
    sentences = ["John lives in New York.", "I work at OpenAI."]
    encoding = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Example labels (dummy data)
    labels = torch.tensor([[1, 2, 3, 4, 0,0,0,0], [1, 5, 6, 7, 0,0,0,0]])  # Replace with your label IDs

    # Forward pass
    model.train()
    output = model(input_ids, attention_mask, labels=labels)
    loss = output.get("loss")
    print("Training loss:", loss.item())

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        predictions = output.get("predictions")
        print("Predictions:", predictions)
