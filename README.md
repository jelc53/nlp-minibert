# CS 224N Default Final Project - Multitask BERT


This is the code for the default final project of Julian Cooper, Thomas Brink, and Quinn Hollister for the Stanford CS 224N class. You can find the handout [here](https://web.stanford.edu/class/cs224n/project/default-final-project-bert-handout.pdf). Our final project report can be found in this repository (CS224N_Final_Report.pdf) as well. For some general instructions (from the CS224N course), see below.

In this project, you will implement some important components of the BERT model to better understanding its architecture. 
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

In our project, we implemented 3 extensions; i) we incorporated regularization in our fine-tuning loss function and parameter gradient step (SMART), ii) we deployed various methods for multitask learning on the three downstream tasks, and iii) we developed rich relational layers that exploit similarity between tasks to stimulate learning in the fine-tuning phase.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, external libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).

## Handout

Please refer to the handout for a through description of the project and its parts.

### Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
