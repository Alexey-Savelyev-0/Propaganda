#from ollama import chat
#from ollama import ChatResponse
import identification
import torch
import os
from pydantic import BaseModel, ValidationError, model_validator
import classification
from typing import List, Tuple
from pathlib import Path
from shutil import copyfile
from enum import Enum
PREDICT = False
class Technique(str, Enum):
    LOADED_LANGUAGE = 'Loaded_Language'
    NAME_CALLING_LABELING = 'Name_Calling,Labeling'
    REPETITION = "Repetition"
    EXAGGERATION_MINIMIZATION = "Exaggeration,Minimisation"
    DOUBT = "Doubt"
    APPEAL_TO_FEAR_PREJUDICE = 'Appeal_to_fear-prejudice'
    FLAG_WAVING = "Flag-Waving"
    CAUSAL_OVERSIMPLIFICATION = 'Causal_Oversimplification'
    SLOGANS = "Slogans"
    APPEAL_TO_AUTHORITY = 'Appeal_to_Authority'
    BLACK_AND_WHITE_FALLACY = 'Black-and-White_Fallacy'
    THOUGHT_TERMINATING_CLICHES = 'Thought-terminating_Cliches'
    BANDWAGON_REDUCTIO_AD_HITLERUM = 'Bandwagon,Reductio_ad_hitlerum'
    WHATABOUTISM_STRAW_MEN_RED_HERRING = "Whataboutism,Straw_Men,Red_Herring"

class Propaganda(BaseModel):
    techniques: List[Technique]
def getTechniqueData():
  articles, article_ids = classification.read_articles("train-articles")
  articles  = articles[300:]
  article_ids = article_ids[300:]
  spans, techniques = classification.read_spans()
  spans = spans[300:]
  no_defs = (
      "You are a propaganda-technique tagger.\n"
      "For each span in the article return only 1 technique. If the spans contain the same technique return it multiple times. "
      "exactly as:\n"
      "<technique-name>\n"
      "It is absolutely vital that you do not output anything else.\n"
      "The techniques are\n:"
      " Loaded Language\n"
"Name Calling,Labeling\n"
"Repetition\n"
"Exaggeration,Minimization\n"
"Doubt\n"
"Appeal to fear-prejudice\n"
"Flag-Waving\n"
"Causal Oversimplification\n"
"Slogans\n"
"Appeal to Authority\n"
"Black-and-White Fallacy\n"
"Thought-terminating Cliches\n"
"Bandwagon,Reductio ad Hitlerum\n"
"Whataboutism,Straw_Men,Red_Herring\n"
  )
  
  with open("LLM_tc_predictions_text_zero_shot.tsv", "w", encoding="utf-8") as out_file:      # utf-8 is safest :contentReference[oaicite:0]{index=0}
      for art_id, article,article_spans in zip(article_ids, articles,spans):
        #article_spans_text = str(article_spans) 
        direct_spans = []
        for s in article_spans:
            direct_spans.append(article[s[0]:s[1]])
        
        response = chat(
              model="deepseek-r1",
              messages=[
                  {"role": "system", "content": no_defs},
                  {"role": "user",   "content": article},
                  {"role": "user",   "content": direct_spans},
              ],
              format=Propaganda.model_json_schema()
            )


        propaganda_techniques = Propaganda.model_validate_json(response.message.content)
        print(propaganda_techniques)
        print(direct_spans)
        for i,span in enumerate(article_spans):
                if i < len(propaganda_techniques.techniques):
                  technique = propaganda_techniques.techniques[i].value
                  out_file.write(f"{art_id}\t{technique}\t{span[0]}\t{span[1]}\n")
                else:
                    # default to one of the classes
                    out_file.write(f"{art_id}\tLoaded_Language\t{span[0]}\t{span[1]}\n")


def getTechniqueData2():
  articles, article_ids = classification.read_articles("train-articles")
  articles  = articles[300:]
  article_ids = article_ids[300:]
  spans, techniques = classification.read_spans()
  spans = spans[300:]
  no_defs = (
      "You are a propaganda-technique tagger.\n"
      "For each span in the article return the one most fitting technique. If the spans contain the same technique return it multiple times. "
      "exactly as:\n"
      "<technique-name>\n"
      "It is absolutely vital that you do not output anything else.\n"
      "The techniques are\n:"
      " Loaded Language\n"
"Name Calling,Labeling\n"
"Repetition\n"
"Exaggeration,Minimization\n"
"Doubt\n"
"Appeal to fear-prejudice\n"
"Flag-Waving\n"
"Causal Oversimplification\n"
"Slogans\n"
"Appeal to Authority\n"
"Black-and-White Fallacy\n"
"Thought-terminating Cliches\n"
"Bandwagon,Reductio ad Hitlerum\n"
"Whataboutism,Straw_Men,Red_Herring\n"
  )
  
  with open("LLM_tc_predictions_text_zero_shot_rf.tsv", "w", encoding="utf-8") as out_file:      # utf-8 is safest :contentReference[oaicite:0]{index=0}
      for art_id, article,article_spans in zip(article_ids, articles,spans):
        #article_spans_text = str(article_spans) 
        direct_spans = []
        output_techniques = []
        for s in article_spans:
            direct_spans.append(article[s[0]:s[1]])
        for direct_span in direct_spans:
          response = chat(
                model="deepseek-r1",
                messages=[
                    {"role": "system", "content": no_defs},
                    {"role": "user",   "content": article},
                    {"role": "user",   "content": direct_span},
                ],
                format=Propaganda.model_json_schema()
              )


          propaganda_techniques = Propaganda.model_validate_json(response.message.content)
          print(propaganda_techniques)
          print(direct_span)
          if len(propaganda_techniques.techniques) > 0:
            output_techniques.append(propaganda_techniques.techniques[0].value)
          else:
            output_techniques.append("Loaded_Language")
        for i,span in enumerate(article_spans):
                if i < len(output_techniques):
                  technique = output_techniques[i]
                  out_file.write(f"{art_id}\t{technique}\t{span[0]}\t{span[1]}\n")
                else:
                    # default to one of the classes
                    out_file.write(f"{art_id}\tLoaded_Language\t{span[0]}\t{span[1]}\n")

if __name__ == "__main__":
    
    if PREDICT:
        getTechniqueData2()
    else:
        src = Path(identification.home_dir) / "architectures"/"tools" / "task-TC_scorer.py"
        dst = Path("predictions") / "task-TC_scorer.py"
        output_file = Path(identification.home_dir) / "predictions" / "merged_task2_bert_labels.txt"
        label_dir = Path(identification.data_dir) / "train-labels-task2-technique-classification"
        articles, article_ids = identification.read_articles("train-articles")
        article_ids = article_ids[300:]
        with open(output_file, "w", encoding="utf-8") as outfile:
          for id in article_ids:
              filename = "article" + id + ".task2-TC.labels"
              filename = f"article{id}.task2-TC.labels"
              file_path = label_dir / filename
              if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)
              else:
                print(f"Warning: {file_path} does not exist and will be skipped.")
        os.system("python3 architectures/tools/task-TC_scorer.py -s predictions/LLM_tc_predictions_zero_shot.tsv -r predictions/merged_task2_labels.txt -p architectures/tools/data/propaganda-techniques-names-semeval2020task11.txt")
        #os.remove( output_file)


