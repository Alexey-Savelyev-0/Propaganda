import os
import torch
import torch.nn as nn
import classification
import torch.nn.functional as F
class Ensemble(nn.Module):
    def __init__(self, model1: nn.Module, model2: nn.Module, weights: tuple = (0.5, 0.5)):
        """
        Ensemble model combining two classifiers using soft voting.

        Args:
            model1 (nn.Module): First pre-trained model.
            model2 (nn.Module): Second pre-trained model.
            weights (tuple): Weights for averaging the probabilities from each model.
                             Default is equal weighting.
        """
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.weights = weights

        # Ensure models are in evaluation mode
        self.model1.eval()
        self.model2.eval()
    def forward(self, b_input_ids, attention_mask=None, token_type_ids=None):
      """
      Forward pass through the ensemble model.

      Args:
          b_input_ids (torch.Tensor): Input IDs tensor.
          b_input_mask (torch.Tensor, optional): Attention mask tensor.
          label_ids (torch.Tensor, optional): Labels tensor.

      Returns:
          torch.Tensor: Predicted class labels if label_ids is None; otherwise, averaged probabilities.
      """
      with torch.no_grad():
        # Obtain logits from each model
        logits1 = self.model1(b_input_ids, attention_mask=attention_mask).logits
        logits2 = self.model2(b_input_ids, attention_mask=attention_mask).logits

        # Convert logits to probabilities
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)

        avg_probs = self.weights[0] * probs1 + self.weights[1] * probs2
        return avg_probs

articles, article_ids = classification.read_articles("train-articles")
spans, techniques = classification.read_spans()
test_articles,test_ids, test_techniques, test_spans = articles[300:], article_ids[300:], techniques[300:], spans[300:]


test_dataloader = classification.get_data(test_articles, test_spans, test_techniques, shuffle=False)

model_path1 = os.path.join(classification.model_dir,"standard_tc2.pt")
model1 = torch.load(model_path1,weights_only=False)
model_path2 = os.path.join(classification.model_dir,"classification_model_reweighed_2025-05-14 00:18:40.280178.pt")
model2 = torch.load(model_path2,weights_only=False)
model = Ensemble(model1,model2)
if classification.device == torch.device("cpu"):
  print("Using CPU for prediction")
  model = torch.load(model_path1, map_location={'cuda:0':'cpu'})

classification.get_dev_predictions(model, test_dataloader,test_ids)
