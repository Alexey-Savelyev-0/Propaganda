import os
import torch
from transformers import BertTokenizerFast, RobertaForTokenClassification, RobertaModel
import transformers
home_dir = "/dcs/22/u2211596/3_UG/3rd_Year_Project/main/Propaganda"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "tc_models")
# distinct_techniques = list(set([y for x in techniques for y in x])) # idx to tag
distinct_techniques = [
 'Flag-Waving',
 'Name_Calling,Labeling',
 'Causal_Oversimplification',
 'Loaded_Language',
 'Appeal_to_Authority',
 'Slogans',
 'Appeal_to_fear-prejudice',
 'Exaggeration,Minimisation',
 'Bandwagon,Reductio_ad_hitlerum',
 'Thought-terminating_Cliches',
 'Repetition',
 'Black-and-White_Fallacy',
 'Whataboutism,Straw_Men,Red_Herring',
 'Doubt'
]
#distinct_techniques.insert(0, 'Non_propaganda')
tag2idx = {t: i for i, t in enumerate(distinct_techniques)}

BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUM_ARTICLES = 300
LANGUAGE_MODEL = "RoBERTa"
REWEIGHING = False
from transformers import RobertaTokenizer
if LANGUAGE_MODEL == "RoBERTa":
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base', lower_case=True)
  #model = RobertaModel.from_pretrained('roberta-base', num_labels=len(tag2idx))
elif LANGUAGE_MODEL == "BERT":
  from transformers import BertTokenizer
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', lower_case=True)
  #model = transformers.BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)

