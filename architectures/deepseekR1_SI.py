#from ollama import chat
#from ollama import ChatResponse
import identification
import torch
import os
from pydantic import BaseModel
from shutil import copyfile
from typing import List, Tuple
from pathlib import Path
PREDICT = False

class Propaganda(BaseModel):
    spans: List[Tuple[int,int]]
def getSpanData():
  articles, article_ids = identification.read_articles("train-articles")
  articles    = articles[300:]
  article_ids = article_ids[300:]
  spans = identification.read_spans()
  SYSTEM_TMPL = (
      "You are a propaganda-span tagger.\n"
      "For the article output each span on its own line, "
      "exactly as:\n"
      "<start_char_position><TAB><end_char_position>\n"
      "It is absolutely vital that you do not output anything else.\n"
      "an example output is:\n" 
      " 265  323\n"
      " 555  666\n"
      " 789 924\n"
      " 1101  1200\n"
      
  )
  SYSTEM_TMPL = (
      "You are a propaganda-span tagger.\n"
      "For the article output each span on its own line, "
      "exactly as:\n"
      "<start_char_position><TAB><end_char_position>\n"
      "It is absolutely vital that you do not output anything else.\n"
      "an example is:"
      "f{articles[299]}\n"
      "f{spans[299]}\n"
  )
  
  with open("LLM_predictions_one_shot.tsv", "w", encoding="utf-8") as out_file:      # utf-8 is safest :contentReference[oaicite:0]{index=0}
      for art_id, article in zip(article_ids, articles):

        response = chat(
              model="deepseek-r1",
              messages=[
                  {"role": "system", "content": SYSTEM_TMPL},
                  {"role": "user",   "content": article},
              ],
              format=Propaganda.model_json_schema()
            )

          # Split model output into individual span lines
          #for span in resp["message"]["content"].strip().splitlines():
              #if span:                                            # skip any blank lines
                  #out_file.write(f"{art_id}\t{span}\n")
        
        propaganda_span = Propaganda.model_validate_json(response.message.content)
        print(propaganda_span)
        for start, end in propaganda_span.spans:
                out_file.write(f"{art_id}\t{start}\t{end}\n")

if __name__ == "__main__":
    if PREDICT:
        getSpanData()
    else:
        src = Path(identification.home_dir) / "architectures"/"tools" / "task-SI_scorer.py"
        dst = Path("predictions") / "task-SI_scorer.py"
        articles, article_ids = identification.read_articles("train-articles")
        article_ids = article_ids[300:]
        for id in article_ids:
            filename = "article" + id + ".task1-SI.labels"
            copyfile(os.path.join(identification.data_dir, "train-labels-task1-span-identification/" + filename), os.path.join(identification.home_dir,"predictions/" + filename))
        # change name of second file to get score
        os.system("python3 predictions/task-SI_scorer.py -s predictions/LLM_si_predictions_one_shot.tsv -r predictions/ -m")
        for id in article_ids:
            filename = "article" + id + ".task1-SI.labels"
            os.remove(os.path.join(identification.home_dir,"predictions/" + filename))
