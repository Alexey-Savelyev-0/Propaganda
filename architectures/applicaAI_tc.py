import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer


class APPLICA_TC_1(nn.Module):
    def __init__(self, num_labels=15, ):
        super(APPLICA_TC_1, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        hidden_size = self.roberta.config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels=None,token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss
        return logits
    

class APPLICA_TC_2(nn.Module):
    def __init__(self, num_labels):
        super(APPLICA_TC_2, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
    
    def custom_loss_fn(self, logits, labels):
        # Placeholder for custom loss function
        loss = WeightedBinaryCrossEntropy(logits,labels)  # Replace with actual loss computation
        return loss
    
    def forward(self, input_ids, attention_mask, labels=None,token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        
        if labels is not None:
            loss = self.custom_loss_fn(logits, labels)
            return logits, loss
        return logits
    

def compute_class_weights(class_frequencies):
    """
    Compute class weights based on frequency.
    :param class_frequencies: A tensor of shape (num_classes,) with class frequencies.
    :return: A tensor of class weights of the same shape.
    """
    max_freq = class_frequencies.max()
    class_weights = max_freq / class_frequencies
    return class_weights


class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, class_frequencies):
        """
        :param class_frequencies: A tensor of shape (num_classes,) containing class frequencies.
        """
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.class_weights = compute_class_weights(class_frequencies)

    def forward(self, x, y):
        """
        :param x: Predictions of shape (batch_size, num_classes) after sigmoid.
        :param y: Multi-hot encoded labels of shape (batch_size, num_classes).
        :return: Weighted BCE loss.
        """
        batch_size, num_classes = x.shape
        
        # Compute binary cross-entropy per element
        loss = - (self.class_weights * y * torch.log(x) + (1 - y) * torch.log(1 - x))
        
        # Average over all elements
        return loss.mean()




class EnsembleModel(nn.Module):
    def __init__(self, num_labels):
        super(EnsembleModel, self).__init__()
        self.model1 = APPLICA_TC_1(num_labels)
        self.model2 = APPLICA_TC_2(num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None,token_type_ids=None):
        logits1 = self.model1(input_ids, attention_mask,token_type_ids)
        logits2 = self.model2(input_ids, attention_mask,token_type_ids)
        
        # Average the logits
        avg_logits = (logits1 + logits2) / 2
        
        if labels is not None:
            loss1 = self.model1.loss_fn(logits1, labels)
            loss2 = self.model2.custom_loss_fn(logits2, labels)
            avg_loss = (loss1 + loss2) / 2
            return avg_logits, avg_loss
        
        return avg_logits