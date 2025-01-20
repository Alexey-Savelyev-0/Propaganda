import os
import sys
BATCH_SIZE = 8
NUM_ARTICLES = 10
TAGGING_SCHEME = "BIOE"
#LANGUAGE_MODEL:= "BERT"| "RoBERTa"
LANGUAGE_MODEL = "RoBERTa"
## if language model is RoBERTa use that
if LANGUAGE_MODEL == "RoBERTa":
  from transformers import RobertaTokenizer
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base', lower_case=True)
else:
  from transformers import BertTokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', lower_case=True)

home_dir = "C:/CS/3rd_Year_Project/Propaganda-detection-experiments/main"
data_dir= r"C:\\CS\\3rd_Year_Project\\Propaganda-detection-experiments\\main\\datasets"
model_dir=  r"C:\\CS\\3rd_Year_Project\\Propaganda-detection-experiments\\main\\applicaAI\\model_dir"
#data_dir = os.path.join(home_dir, "\datasets")
print(data_dir)



#model_dir = os.path.join(home_dir, "\model_dir")
if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
