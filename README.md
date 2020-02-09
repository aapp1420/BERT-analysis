# BERT-analysis
This is the homework repository of Deep Learning for Human Language Processing 
## Goal
The goal for this homework is:
* train a chinese nli task and save model 
* generate dataset which created from each layer embedding of each data in xnli sample data (two dataset: one is created from pretrained model, the other one is fintuned model.)
* analysis embeddings (anisotropy, self-similarity, Maximum Eigenvalue Variances) (adjusted version)

## Dataset: Xnli zh
Training:  https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip  

(Use XNLI-MT-1.0/multinli/multinli.train.zh.tsv)

Testing:  https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip 

(Use XNLI-1.0/xnli.dev.tsv)

replace the same name folder in the dataset folder.

Analysis Data: https://drive.google.com/open?id=1B-J6iuSj4OX_o3igmofUBPpY8lG9IVIT 

(xnli.examples.tsv)


## Step 1 Train xnli and achieve accuracy performance approximately 73-76%
Hint: Utilize run_xnli.py

* Train a model on multinli.train.zh.tsv and test on xnli.dev.tsv
* Plot the training loss and testing accuracy 
* 3 epochs may be sufficient
* Remember to save the best model for later analysis

## Step 2 Generate pretrained data and finetune data for xnli-sample data
Example Code: generate-similarity-data.py

store each data: with its input_ids/ layer_0_embedding / layer_1_embedding / layer_2_embedding / layer_3_embedding ... layer_12_embedding

Hint: you need to save the data generate from pretrained model and fintuned-model (the model you save)
## Step 3 analysis bert layer embedding:

