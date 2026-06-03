
## Table of Contents
- [Introduction](#Introduction)
- [Theory](#Theory)
- [Implementation](#Implementation)
- [Results and Evaluation](#Results)
  


# 1. Introduction
My third year dissertation project at the University of Warwick. It is best summarised as an ablation study & test harness of the top performing models for Computational Propaganda Detection. This document summarises the methodology, structure, and outcome of the project. For details on the survey, linguistic theory, and other background research done for this task, please refer to [[this document here]](./Final_Report.pdf)
## 1.1 Defining Propaganda
Propaganda is pervasive in digital media, often shaping the opinions of its consumer in subtle ways. This slight manipulation may prove to be highly lucrative for the propagandist, with benefits ranging from improving their perceived public image, to shaping support for an important cause. Conversely, the consumer is affected by propaganda in a manner which they are unlikely to be comfortable with, knowing the intention of the author and the information which had been kept away from them. As such, it is in the consumer's best interest to be aware of the biases present in the media they consume.
This project therefore discusses the various forms digital propaganda may take form, previous campaigns that were ran, and the trajectory of propaganda in the 21st century. It also evaluates various methodologies for detecting digital propaganda, with tests ran and models developed on some of the most promising approaches.

# 2. Task Discusison
During the survey part of the project, many approaches were considered for computational propaganda detection, with most of them falling into 2 categories; Network Analysis and Text Classification. An approach was chosen from the latter category. This is primarily for 2 reasons:  
1. Graph/Network Analysis struggles with a lack of relevant, real-world data - misinformation campaigns done on these platforms are inherently difficult to track, and are propriatary information.
2. Graph/Network Analysis is significantly less effective. This is due agent evolution - as detection methods improve, the agents adapt in response. This is less relevant for text analysis which is less definitely identifies a bad actor, and looks at the persuasiveness of the text.  
Less importantly, this project was educational, and allowed me to learn NLP through application. As such the text-based approaches were slightly more appealing.
## 2.1 Task Definition 
The pipeline is comprised of a a 2 stage sequence - __Span Identification (SI)__ and __Technique Classification (TC)__. of persuasive techniques in text. To briefly explain their purpose, Span identification outlines the range of the propaganda in text, and Technique Classification classifies the rhetoric device being used in that range.

Detecting propaganda manually is already challenging, vague, and requires expertise knowledge - doing so automatically is signifcantly harder. This approach circumvents some of the challenge by instead opting to detect the use of persuasion in text, with the implication being that text relying on heavy uses of persuasion, rather than fact, is likely to be propaganda. This assumption is somewhat flawed as persuasion!=propaganda, but still serves as a useful metric for healthier consumption of media by encouraging critical analysis of the biases of a text. Unfortunately, as of writing this ( Jan 2026) this is as far as propaganda detection pipelines have gotten in research (to the best of my knowledge).
## 2.2 Identification
Propaganda may be identified in text by performing some level of classification of individual sub-components on the text - the length of the sub-component depends on both the setup and the training data provided to the models. For the sake of this task, the level which we concern ourselves with is the word-wise level - by classifying individual words as belonging to the propaganda class, we may create 'spans' of it in text. These are meant to be equivalent to rhetoric techniques used in SemEval 2020 Task 11. As an example, in the sentence:
 <br />
_"Despite pervasive beliefs, there is no war in Ba-Sing-Se"_,
 <br /> and if there **is**, in fact, a war in Ba-Sing-Se, then the model will likely choose to classify the words 4-9 as a propaganda snippet.

## 2.3 Classification
Once a a propaganda span is identified, we then choose to classify it as one of many rhetoric techniques. <br />
1. Loaded language - Using words/phrases with strong emotional implications [
2. Name calling or labelling - labelling the the target of the propaganda as something the consumer
dislikes, or has negative associations.
3. Repetition - the repetition of the desired message.
4. Exaggeration or minimization - exaggerating a given action to make it seem more larger, serious, or
in any other terms more intense that it actually is. Minimization is the opposite - making a given
topic seem less significant that it may actually be.
5. Doubt - questioning the credibility of an entity or statement, in a critical, non-constructive manner.
6. Appeal to fear/prejudice - seeking to build support for an idea by instilling anxiety and panic in
the population towards an alternative, possibly based on preconceived judgments.
7. Flag-waving - laying on a strong national feeling (or any other large group the consumer may
identify themselves with) to justify or promote an action or idea.
8. Causal oversimplification - reducing a complicated set of circumstances to one cause for a particular
issue.
9. Slogans - a brief and striking phrase that may include labelling and stereotyping.
10. Appeal to authority - Stating that a claim is true simply because a valid authority or expert on the
issue supports it, without any other supporting evidence.
11. Black-and-white fallacy, dictatorship - Presenting two alternative options as the only possibilities,
when in fact more possibilities exist.
12. Thought-terminating cliche - Words or phrases that discourage critical thought and meaningful
discussion about a given topic.
13. Whataboutism - discrediting an opponent’s position by charging them with hypocrisy without di-
rectly disproving their argument.
14. Reductio ad Hitlerum - persuading an audience to disapprove an action or idea by suggesting that
the idea is popular with groups despised by the target audience. As the name suggests, a common
example of the despised group is the Nazi Party.
15. Red herring - Introducing irrelevant material to the issue being discussed, so that everyone’s atten-
tion is diverted away from the points made.
16. Bandwagon - attempting to persuade the target audience to join in and take the course of action
because “everyone else is taking the same action”.
17. Obfuscation, intentional vagueness, or confusion - using deliberately unclear words, so that the
audience may have its own interpretation.
18. Straw man - when an opponent’s proposition is substituted with a similar one which is then refuted
in place of the original.
<br/>


Each snippet is assigned a single technique which suits it best (note that snippets may overlap and contain one another), in which case they'd be assigned one technique each. 

This is where the core pipeline ends. These snippets can be useful if displayed, or used to train another model for further classification. We avoid assigning an explicit propaganda label due to a number of technical limits and ethical issues. Namely, to the best of my knowledge, there exists no dataset where techniques are labelled in the way they're labelled in the pipeline, **and** the whole article is labelled as propaganda. As such, there's no way to accurately train the model in a way that wouldn't heavily reflect the author's biases (in this case, my bias). Additionally, by exposing a heavy use of rhetoric devices, the pipeline encourages the recipient to engage with the literature critically, and decide for themselves whether it is propaganda. An outright propaganda label may have the opposite effect, where instead of some source being blindly treated as fact, the model's verdict is. Of course, this is a double-edged sword - a lack of decisive label from the model may mean it's not particularly useful. I believe it is preferrable that a pipeline such as this is overly causious rather than underly - labelling too many things as propaganda causes more damage than not labelling enough.


# 3. Implementation
Below is a description of the architectures used for this task. <br/>
The base version of the model can be described as a token-wise classification task using BIO classes, followed by sentence-wide classifications into one of the 18 listed classes. These classification models can vary in terms of complexity - the most simple ones involve a BERT model followed by a classification layer. The model can however be improved via various additions, such as __Multi-Task Learning__, and manual token enrichment.

<img width="611" height="1141" alt="1ff91b8d65f96ffe149d1dc69f33209c4720af32" src="https://github.com/user-attachments/assets/07cb1720-f1c8-47d5-ae7a-82526abc7b68" />

The classification task is more straightforward - it singly relies on a BERT based classifier.

# 4. Results
## 4.1 Span Identification
<img width="577" height="110" alt="image" src="https://github.com/user-attachments/assets/e75acebb-3019-4e00-b568-be6cdd916bde" />

<img width="732" height="97" alt="image" src="https://github.com/user-attachments/assets/1cecaa71-3c72-4446-8574-aa59d6b687f3" />


## 4.2 Task Classification

<img width="439" height="340" alt="image" src="https://github.com/user-attachments/assets/7d60654d-bf9b-40e3-a373-eb619cd47e2e" />

# 5. Reflection
