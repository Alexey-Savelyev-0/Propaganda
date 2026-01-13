import os
import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, RobertaModel
import transformers

BATCH_SIZE = 16
NUM_ARTICLES = 300
TAGGING_SCHEME = "BIO"
LANGUAGE_MODEL = "RoBERTa"
EPOCHS = 5
TLC = True
SLC = True
SAVE_MODEL = True
SELF_TRAIN = False
LEARNING_RATE = 1e-5
POS = True
NER = True
NELA = False
CRF = True


distinct_techniques = {
 'Flag-Waving':0,
 'Name_Calling,Labeling':1,
 'Causal_Oversimplification':2,
 'Loaded_Language':3,
 'Appeal_to_Authority':4,
 'Slogans':5,
 'Appeal_to_fear-prejudice':6,
 'Exaggeration,Minimisation':7,
 'Bandwagon,Reductio_ad_hitlerum':8,
 'Thought-terminating_Cliches':9,
 'Repetition':10,
 'Black-and-White_Fallacy':11,
 'Whataboutism,Straw_Men,Red_Herring':12,
 'Doubt':13,
  0:-1
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if LANGUAGE_MODEL == "RoBERTa":
  tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', lower_case=True)
  #model = RobertaModel.from_pretrained('roberta-base')
elif LANGUAGE_MODEL == "BERT":
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', lower_case=True)
  model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
  sentence_len = 256

else:
  raise Exception("Unknown LLM For Hitachi SI detected")







home_dir = "C:/CS/3rd_Year_Project/Propaganda-detection-experiments/main"
home_dir = "/dcs/22/u2211596/3_UG/3rd_Year_Project/main/Propaganda"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "si_models")


if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
