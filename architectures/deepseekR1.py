from ollama import chat
from ollama import ChatResponse
import identification
import torch
import os
import identification.hitachi_utils as hitachi_utils




response: ChatResponse = chat(model='deepseek-r1', messages=[
  {
    'role': 'user',
    'content': 
      '''You are a propaganda expert tasked with identifying the spans of propaganda for a given text.
        For any article, you will give the article-wide index of the start and end characters for each propaganda 
        span in the format (start, end). There may be multiple spans in a single text, and they may overlap.
        Please provide the labels for the following text:
        "{article}"
    ''',
  },
])


articles, article_ids = identification.read_articles('train-articles')
spans = identification.read_spans()


#print(response['message']['content'])
# or access fields directly from the response object 
print(response.message.content)