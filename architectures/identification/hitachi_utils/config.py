import os
import torch
from transformers import BertTokenizerFast
import transformers

"""torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping"""

BATCH_SIZE = 4
NUM_ARTICLES = 2
TAGGING_SCHEME = "BIO"
LANGUAGE_MODEL = "BERT"
LEARNING_RATE = 3.5e-4
EPOCHS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if LANGUAGE_MODEL == "RoBERTa" or LANGUAGE_MODEL == "RoBERTa-CRF":
  from transformers import RobertaTokenizer
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base', lower_case=True)
else:
  from transformers import BertTokenizer
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', lower_case=True)
  model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
  sentence_len = 256















home_dir = "C:/CS/3rd_Year_Project/Propaganda-detection-experiments/main"
data_dir= r"C:\\CS\\3rd_Year_Project\\Propaganda-detection-experiments\\main\\datasets"
model_dir=  r"C:\\CS\\3rd_Year_Project\\Propaganda-detection-experiments\\main\\architectures\\model_dir"


if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
