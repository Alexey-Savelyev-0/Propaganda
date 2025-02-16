import os
import sys
BATCH_SIZE = 2
NUM_ARTICLES = 5
TAGGING_SCHEME = "BIO"
#LANGUAGE_MODEL:= "BERT"| "RoBERTa" | "RoBERTa-CRF" | "DeBERTa" | "DeBERTa-CRF"
LANGUAGE_MODEL = "BERT"






if LANGUAGE_MODEL == "RoBERTa" or LANGUAGE_MODEL == "RoBERTa-CRF":
  from transformers import RobertaTokenizer
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base', lower_case=True)
else:
  from transformers import BertTokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', lower_case=True)















home_dir = "C:/CS/3rd_Year_Project/Propaganda-detection-experiments/main"
data_dir= r"C:\\CS\\3rd_Year_Project\\Propaganda-detection-experiments\\main\\datasets"
model_dir=  r"C:\\CS\\3rd_Year_Project\\Propaganda-detection-experiments\\main\\architectures\\model_dir"


if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
