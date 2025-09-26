# Summary
An ablation study of the top performing models for Computational Propaganda Detection in SemEval2020 Task 11. 
The task is seperated into 2 subtasks - __Span Identification (SI)__ and __Technique Classification (TC)__. Span identification outlines the range of the propaganda in text, Technique Classification classifies the rhetoric device being used. For both tasks, various models are developed and compared in a seperate paper. 

# Task
The files in this repository can be considered to belong to one of two studies - classification and identification. Both studies are on sub-tasks required for the pipeline of computational propagranda detection to function. These are briefly explained below.
## Identification
Propaganda may be identified in text by performing some level of classification of individual sub-components on the text - the length of the sub-component depends on both the setup and the training data provided to the models. For the sake of this task, the level which we concern ourselves with is the word-wise level - by classifying individual words as belonging to the propaganda class, we may create 'spans' of it in text. These are meant to be equivalent to rhetoric techniques used in SemEval 2020 Task 11. As an example, in the sentence:
 <br />
_"Despite pervasive beliefs, there is no war in Ba-Sing-Se"_,
 <br /> the model will likely choose to classify the words 4-9 as one propaganda snippet.

## Classification
Once a a propaganda span is identified, we then choose to classify it as one of the many techniques in the training data. <br />
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
Note that this list is chosen due to the available training data - the specific combiniation of techniques isn't particularly special in any meaningful manner (beyond being grounded in sound literature). As such this list may have room for improvement (some sources list a number of individual rhetoric techniques in the 80s), or conversely the user may find many of these techniques to be redundant.

Each snippet is assigned a single technique which suits it best (note that snippets may overlap and contain one another). This is where both this code and the original task end the work - these classified snippets may be used for educational purposes, or to inspire critical thinking during the consumption of a specific fed text. An expansion on the latter use-case would be necessary for convenience and usability - for instance, a web-extension.

# Architecture
Below is a description of the architectures used for this task.
