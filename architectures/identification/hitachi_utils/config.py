import os
import torch
from transformers import BertTokenizerFast
import transformers


BATCH_SIZE = 4
NUM_ARTICLES = 2
TAGGING_SCHEME = "BIO"
LANGUAGE_MODEL = "BERT"
LEARNING_RATE = 3.5e-4
EPOCHS = 4
SLTC = False
SLC = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if LANGUAGE_MODEL == "RoBERTa":
  from transformers import RobertaTokenizer
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base', lower_case=True)
if LANGUAGE_MODEL == "BERT":
  from transformers import BertTokenizer
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', lower_case=True)
  model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
  sentence_len = 256

else:
  raise Exception("Unknown LLM For Hitachi SI detected")
















#data_dir= r"\\dcs\\22\\u2211596\\3_UG\\3rd_Year_Project\\main\\Propaganda\\datasets"
#model_dir=  r"\\dcs\\22\\u2211596\\3_UG\\3rd_Year_Project\\main\\Propaganda\\architectures\\model_dir"
home_dir = "/dcs/22/u2211596/3_UG/3rd_Year_Project/main/Propaganda"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "architectures", "model_dir")


if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
