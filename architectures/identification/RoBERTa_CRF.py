from torchcrf import CRF
import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaForTokenClassification
""" supposedly already inherently used vetebri loss - confirm with own research."""


class RobertaCRF(nn.Module):
    def __init__(self, bert,num_labels,weights = None, **kwargs):
        super(RobertaCRF, self).__init__()
        #self.roberta = bert
        dropout = kwargs.get("dropout", 0.1)
        attention_dropout = kwargs.get("attention_dropout", 0.1)
        self.use_crf = True
        self.roberta = RobertaModel.from_pretrained(
            'roberta-base',
            hidden_dropout_prob=dropout, 
            attention_probs_dropout_prob=attention_dropout
        )

        hidden_size = self.roberta.config.hidden_size
        print(hidden_size)
        # Linear layer for converting RoBERTa output to label space
        self.fc = nn.Linear(hidden_size, num_labels)

        self.dropout = nn.Dropout(dropout)

        # CRF layer for sequence labeling
        self.crf = CRF(num_labels, batch_first=True)

        self.weights = weights.to("cuda") if (weights is not None and torch.cuda.is_available()) else weights

        #self._enforce_transition_constraints()


    def _enforce_transition_constraints(self):
        """
        Artificially set constraints on the CRF transition matrix.
        For IOB tagging, we enforce:
        - "I" cannot follow "O".
        - "I" must follow "B" or "I" of the same entity type.
        - "B" can follow any tag.
        """
        num_labels = self.crf.num_tags
        transition_matrix = self.crf.transitions.data
        
        # Initialize all transitions to -inf (disallowed)
        transition_matrix.fill_(-float('inf'))
        
        # Define allowed transitions
        for i in range(num_labels):
            for j in range(num_labels):
                # Allow transitions from "O" to "O" or "B"
                if i == 0:  # Assuming "O" is the first label
                    if j == 0 or j % 2 == 1:  # Allow "O -> O" or "O -> B-*"
                        transition_matrix[i, j] = 0
                
                # Allow transitions from "B" to "I" of the same entity type
                elif i % 2 == 1:  # Assuming "B-*" tags are odd indices
                    if j == i + 1:  # Allow "B-X -> I-X"
                        transition_matrix[i, j] = 0
                
                # Allow transitions from "I" to "I" of the same entity type
                elif i % 2 == 0 and i != 0:  # Assuming "I-*" tags are even indices
                    if j == i:  # Allow "I-X -> I-X"
                        transition_matrix[i, j] = 0
                
                # Allow transitions from any tag to "B"
                if j % 2 == 1:  # Allow "* -> B-*"
                    transition_matrix[i, j] = 0


    def forward(self, input_ids,token_type_ids, attention_mask, labels=None):
        # Pass input through RoBERTa model
        # doesn't the outputs variable already contain the last hidden state?

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

       
        #assert sequence_output.shape[-1] == self.fc.in_features, f"Expected {self.fc.in_features}, got {sequence_output.shape[-1]}"
        # Apply linear layer to get scores for each label
        emissions = self.fc(sequence_output)
        emissions = self.dropout(emissions)
        attention_mask = attention_mask.bool()
        if labels is not None:
            # Truncate labels to match sequence length
            labels = labels[:, :emissions.size(1)]

            if self.use_crf:
                # Compute CRF loss if CRF is enabled
                loss = -self.crf(emissions, labels, mask=attention_mask, reduction='mean')
            else:
                # Compute standard cross-entropy loss if CRF is disabled
                loss_fct = nn.CrossEntropyLoss(weight=self.weights)
                loss = loss_fct(emissions.view(-1, emissions.size(-1)), labels.view(-1))

            return {
                "loss": loss,
                "logits": emissions,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
            }
        else:
            if self.use_crf:
                predictions = self.crf.decode(emissions, mask=attention_mask)
                # Pad predictions to the maximum sequence length in the batch
                predictions = [torch.tensor(pred, device=emissions.device) for pred in predictions]
                predictions = pad_sequence(predictions, batch_first=True, padding_value=-100)
            else:
                # Use argmax for predictions if CRF is disabled
                predictions = torch.argmax(emissions, dim=-1)

            return {
                "logits": emissions,
                "predictions": predictions,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
            }