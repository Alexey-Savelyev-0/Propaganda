import os
import sys
BATCH_SIZE = 2
NUM_ARTICLES = 5
TAGGING_SCHEME = "BIO"
LANGUAGE_MODEL = "BERT"
SELF_TRAINING = False
SELF_TRAINING_ARTICLE_VOLUME = 0





if LANGUAGE_MODEL == "RoBERTa" or LANGUAGE_MODEL == "RoBERTa-CRF":
  from transformers import RobertaTokenizer
  tokenizer = RobertaTokenizer.from_pretrained('roberta-large', lower_case=True)
else:
  from transformers import BertTokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', lower_case=True)













home_dir ="C:/CS/3rd_Year_Project/Propaganda-detection-experiments/main"
#home_dir = "/dcs/22/u2211596/3_UG/3rd_Year_Project/main/Propaganda"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "architectures", "model_dir")


if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
