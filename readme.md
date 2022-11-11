<h2 align="center">Entity Extraction and Categorization</h2>
<p align="center">
  Developed by <a href="https://github.com/ByUnal"> M.Cihat Unal </a> 
</p>

## Overview

The API provides entity extraction algorithm from job postings. Finally, calculates the similarity 
scores between job postings.

## Requirements
Install dependencies.

``
pip install -r requirements.txt
``

Download the following for working with spacy without getting any kinds of error.

``
python -m spacy download en_core_web_sm
``

## Dataset
The [dataset](https://drive.google.com/drive/folders/1e1JA0KiGt9efSqHgzeoRe-msTssbppDt?usp=sharing) has been shared
with me by the link from authorities. You can make an application to the authorities to obtain the data by using the link.
After downloading, put the files under **data** folder.
Run the following code in the project directory. This file includes following:
- Data preprocessing
- Data labelling (by algorithm)
- Data creation for training and test
```
python main.py
```
It will create TRAIN_DATA.spacy file and save it under the TRAIN_DATA folder. We will use this file for the training.

## Training
Use the [link](https://spacy.io/usage/training#quickstart) to create config file for the training. You can specify training
parameters thanks to this guide, which is shared by the link above.  You can copy all the contents or download to 
working directory. In both case, config file's name should be **base_config.cfg**.

Then run the following command:

``
python -m spacy init fill-config base_config.cfg config.cfg
``

A config.cfg file will appear in your working directory.
Next, run the following to begin training:

``
python -m spacy train config.cfg --output ./output
``

After training is complete, the resulting model will appear in a new folder called output.

## Categorize and Calculate Similarity score
The algorithm uses cosine similarity to find similarities between job posts. Moreover, job posts have the 
relevant skills will be awarded while similarity calculation.

Run the following code categorization and to calculate similarity scores
```
python categorize.py
```

Categories and similarity scores between job posts will be extracted to separate CSV files.
You can find these files under the output_files folder after python file executed.