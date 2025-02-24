from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='deepseek-r1', messages=[
  {
    'role': 'user',
    'content': 
      '''You are a propaganda expert tasked with identifying the spans of propaganda for a given text.
        For every word, you must idnetify whether it is propaganda or not. To do so, you will provide a label for each word.
        The labels are:
        - 2: beggnining of propaganda span
        - 1: inside propaganda span
        - 0: outside propaganda span
        As an example, consider the following text:
        "The government has been accused of corruption and nepotism."
        The correct labels for this text are:
        "The" 0
        "government" 0
        "has" 0
        "been" 0
        "accused" 0
        "of" 0
        "corruption" 2
        "and" 1
        "nepotism" 1
        "." 0
        Please provide the labels for the following text:
        "The company has been accused of fraud and embezzlement."
    ''',
  },
])



#print(response['message']['content'])
# or access fields directly from the response object 
print(response.message.content)