import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import spacy

from preprocessing import preprocessing
from calculate_similarities import find_similarity

import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("data/jobpostings.csv")

# Remove all duplicate rows
df = df.drop_duplicates(keep='last')

df_tech = pd.read_excel("data/Technology Skills.xlsx")
tech_skills = list(df_tech["Example"].unique())
tech_skills = [ts.lower() for ts in tech_skills]

df_skill = pd.read_excel("data/Skills.xlsx")
regular_skills = list(df_skill["Element Name"].unique())
regular_skills = [ts.lower() for ts in regular_skills]

train, test = train_test_split(df, test_size=0.2, random_state=14, shuffle=True)
train = train.reset_index()
test = test.reset_index()

# Preprocessing...
print("Preprocessing...")
for idx, row in tqdm(test.iterrows()):
    test['Job Description'][idx] = preprocessing(test['Job Description'][idx])

# load the trained model
nlp_output = spacy.load("output/model-best")

category_list = []

# Categorizing...
print("Categorizing...")
for idx, row in tqdm(test.iterrows()):
    # pass our test instance into the trained pipeline
    doc = nlp_output(test["Job Description"][idx])

    obj = {"Job Id": test["Job Id"][idx], "Entity Values": list(doc.ents)}
    category_list.append(obj)

cols = ['JOB ID', 'Entity Values']
lst = []
for idx in tqdm(range(len(category_list))):
    entities = [str(s) for s in category_list[idx]["Entity Values"]]
    lst.append([category_list[idx]["Job Id"],
                entities])
df_categories = pd.DataFrame(lst, columns=cols)

# Extract categories to CSV file
df_categories.to_csv("output_files/Entity_catagories.csv")


# Calculate similarities
print("Calculating similarities...")
cols = ['JOB ID A', 'JOB ID B', 'Similarity Score']
lst = []
nlp = spacy.load('en_core_web_md')
for idx in range(len(category_list)):
    for i in range(idx + 1, len(category_list)):
        if (len(category_list[idx]["Entity Values"]) == 0) or (len(category_list[i]["Entity Values"]) == 0):
            continue
        score = find_similarity(category_list[idx]["Entity Values"],
                                category_list[i]["Entity Values"],
                                tech_skills,
                                regular_skills,
                                nlp)

        lst.append([category_list[idx]["Job Id"],
                    category_list[idx]["Job Id"],
                    score])
    print(f"{idx}/{len(category_list)}")

df_similarities = pd.DataFrame(lst, columns=cols)

# Normalize similarities
# Define columns to normalize
x = df_similarities.iloc[:, 2:3]

# normalize (put values between 0 and 1) values of last columns only
df = df_similarities.iloc[:, 2:3] = (x - x.min()) / (x.max() - x.min())

# Sort values based on similarities
df_similarities = df_similarities.sort_values(by=['Similarity Score'], ascending=False)

# Extract similarities to CSV file
df_similarities.to_csv("output_files/job_similarities.csv")
