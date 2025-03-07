import os
import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast
import transformers

BATCH_SIZE = 6
NUM_ARTICLES = 5
TAGGING_SCHEME = "BIO"
LANGUAGE_MODEL = "RoBERTa"
LEARNING_RATE = 2.9e-6
EPOCHS = 4
TLC = False
SLC = False
SAVE_MODEL = True
SELF_TRAIN = False

CURRENT_MODEL = "hitachi_si_20250304_060455" + ".pt"








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
  tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
elif LANGUAGE_MODEL == "BERT":
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', lower_case=True)
  model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
  sentence_len = 256

else:
  raise Exception("Unknown LLM For Hitachi SI detected")
















#data_dir= r"\\dcs\\22\\u2211596\\3_UG\\3rd_Year_Project\\main\\Propaganda\\datasets"
#model_dir=  r"\\dcs\\22\\u2211596\\3_UG\\3rd_Year_Project\\main\\Propaganda\\architectures\\model_dir"
home_dir = "C:/CS/3rd_Year_Project/Propaganda-detection-experiments/main"
#home_dir = "/dcs/22/u2211596/3_UG/3rd_Year_Project/main/Propaganda"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "model_dir")


if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
