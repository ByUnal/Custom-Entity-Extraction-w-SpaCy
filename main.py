# library imports
import asyncio
import pandas as pd
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from preprocessing import preprocessing, convert_to_list

import warnings

warnings.filterwarnings('ignore')


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def prepare_training_data(row):
    # Convert string to list
    # Clear the whitespaces at the beginning and the end of each item
    annotated_skills = convert_to_list(row['Skills'])
    if annotated_skills:
        annotated_skills = [s.strip() for s in annotated_skills]

        # Merge skills list with already annotated skills
        merged_skills = skills + annotated_skills

        # shape the training data
        structure_training_data(row['Job Description'], merged_skills)
    else:
        # shape the training data
        structure_training_data(row['Job Description'], skills)


def structure_training_data(text, kw_list):
    entities = []

    # search for instances of keywords within the text (ignoring letter case)
    for kw in kw_list:

        # Check whether kw equals to "R", "J", ...
        try:
            search = re.finditer(f" {kw} ", text, flags=re.IGNORECASE)
        except re.error:
            # To avoid "multiple repeat" error for inputs have special chars. like "c++"", we use re.escape().
            search = re.finditer(re.escape(f" {kw} "), text, flags=re.IGNORECASE)

        # store the start/end character positions
        all_instances = [[m.start(), m.end()] for m in search]

        # if the callable_iterator found matches, create an 'entities' list
        if len(all_instances) > 0:
            for i in all_instances:
                start = i[0]
                end = i[1]
                entities.append((start, end, kw))

    # add any found entities into a JSON format within collective_dict
    if len(entities) > 0:
        results = [text, {"entities": entities}]
        collective_dict['TRAINING_DATA'].append(results)


def create_training_set(TRAIN_DATA):
    db = DocBin()
    for text, annot in tqdm(TRAIN_DATA):
        doc = nlp.make_doc(text)
        ents = []

        # create span objects
        for start, end, label in annot["entities"]:

            span = doc.char_span(start, end, label=label, alignment_mode="contract")

            # skip if the character indices do not map to a valid span
            if span is None:
                # print("Skipping entity.")
                continue
            else:
                ents.append(span)
                # handle erroneous entity annotations by removing them
                try:
                    doc.ents = ents
                except:
                    ents.pop()
        doc.ents = ents

        # pack Doc objects into DocBin
        db.add(doc)
    return db


if __name__ == "__main__":
    # this dictionary will contain all annotated examples
    collective_dict = {'TRAINING_DATA': []}

    # Load data
    df = pd.read_csv("./data/jobpostings.csv")

    # Remove all duplicate rows
    df = df.drop_duplicates(keep='last')

    df_tech = pd.read_excel("data/Technology Skills.xlsx")
    tech_skills = list(df_tech["Example"].unique())

    df_skill = pd.read_excel("data/Skills.xlsx")
    regular_skills = list(df_skill["Element Name"].unique())

    skills = tech_skills + regular_skills

    # Preprocessing...
    print("\nPreprocessing...")
    for idx, row in tqdm(df.iterrows()):
        df['Job Description'][idx] = preprocessing(df['Job Description'][idx])

    train, test = train_test_split(df, test_size=0.2, random_state=14, shuffle=True)
    train = train.reset_index()
    test = test.reset_index()

    print("\nCombine skills for training...")
    for idx, row in tqdm(train[:50000].iterrows()):
        prepare_training_data(row)

    print("\nLength of collective dict: ", len(collective_dict['TRAINING_DATA']))
    # define our training data to TRAIN_DATA
    TRAIN_DATA = collective_dict['TRAINING_DATA']

    # create a blank model
    nlp = spacy.blank('en')

    print("Prepare data for training...")
    TRAIN_DATA_DOC = create_training_set(TRAIN_DATA)

    # Export results (here I add it to a TRAIN_DATA folder within the directory)
    TRAIN_DATA_DOC.to_disk("./TRAIN_DATA/TRAIN_DATA.spacy")
