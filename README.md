An ablation study of the top performing models for Computational Propaganda Detection in SemEval2020 Task 11. 
The models here are the Span Identification and Task Classification of Hitachi and ApplicaAI. Some tweaks to them are added and played around with,
with varying responses. Full details are in the accompanying dissertation.



===-------------------------------------------------------------------------------------------
FOLDER STRUCTURE:

architectures: Contains source code for all of the models tested.
datasets: contains main dataset (SemEval 2020 Training set) for and openwebtext.
predictions: tools provided by Da San Martino et Al. for the joint task.

in architectures are the files predict_tc and predict_si - these can be used to display the F1 scores for both respective tasks, although they may require manual manipulation of filenames depending on which model you wish to test.


The original code is built on top of the submission of NewsSweeper for SemEval 2020 task 11. 
Their primary contribution to this project is the source functions - the models (besides the base BERT cases)
covered in this repository are different. Furthermore, singificant bug fixing had to be done to make their code workable on this system. The main developed models are in hitachi_si.py and BERT_tc.py. There also exist some other unfinished segments of models in the filebase - unless explicitly mentioned in the report, please assume these to be non-functional.


To use - the requirements in requirements.txt must be downloaded. This can be done through:
conda create --name <env> --file requirements.txt while in the main working file. 
Additionally, to run Deepseek R1, Ollama needs to be installed.

NOTE:
model files were too large to submit – as such they can be found and installed here: https://livewarwickac-my.sharepoint.com/:f:/r/personal/u2211596_live_warwick_ac_uk/Documents/dissertation%20span%20identification%20files?csf=1&web=1&e=BwH03v place the si models into the si_models file to use, do the same with the tc models.
