"""
1. step 1 - create a roberta-crf model
2. step 2 - train model on data
3. step 3 - label an existing db using this model.

"""

"""
step 1.1 - create roberta model
"""


from transformers import RobertaTokenizer, RobertaModel
from transformers import Trainer, TrainingArguments
from dataset import *
import torch
from torch import nn
from torchcrf import CRF
from .utils_ner import convert_examples_to_features, get_labels, read_examples_from_file
import spacy
# Step 1: Load the dataset
articles_content, articles_id, propaganda_techniques_names = load_data("datasets/train-articles","propaganda-techniques-names-semeval2020task11.txt")  # Example dataset


# define nlp - change later
nlp = spacy.load("en_core_web_sm")
labels_path = "datasets/train-task1-SI.labels"
train_file_path = "train.tsv"
dev_file_path = "dev.tsv"


train_ids, dev_ids = get_train_dev_files(articles_id, articles_content, nlp, labels_path, train_file_path,
                                                     dev_file_path, True, 0.18, 42)
print(train_ids)



# Step 4: Define the model with a CRF layer
class RobertaWithCRF(nn.Module):
    def __init__(self, num_labels=3):
        super(RobertaWithCRF, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.dropout = nn.Dropout(0.1)
        #self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        # what the fuck does a classifier have anything to do with this
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self,input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        """
        the output of vector needs to be reduced to logits - in this instance, this is done via a linear classifier.
        """
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return predictions

