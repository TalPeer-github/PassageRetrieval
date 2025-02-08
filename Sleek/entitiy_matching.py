
import re
import thefuzz.utils
from thefuzz import process
from thefuzz import fuzz
from utils import persons_ents
import ast
import pandas as pd
import numpy as np
import spacy
import nltk

import warnings

warnings.filterwarnings("ignore")
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

def get_match(person):
    """
    Compare different scorers to choose the best metric
    :param person:
    :return:
    """
    matches_set = process.extractOne(person, persons_ents, scorer=fuzz.token_set_ratio,
                                 processor=thefuzz.process.default_processor)
    matches_sort = process.extractOne(person, persons_ents, scorer=fuzz.partial_token_sort_ratio,
                                 processor=thefuzz.process.default_processor)
    match_score_set = matches_set[1]
    match_score_sort = matches_sort[1]
    # print(f"PERSON:{person}")
    # print(matches_set)
    # print(matches_sort)
    if match_score_sort < 85:
        matches_2 = process.extractOne(person, persons_ents, scorer=fuzz.partial_token_set_ratio,
                                     processor=thefuzz.process.default_processor)
        match_score_2 = matches_2[1]
        if match_score_2 < 80:
            matches_3 = process.extractOne(person, persons_ents, scorer=fuzz.token_sort_ratio,
                                           processor=thefuzz.process.default_processor)
            # print(f"person: {person} matches3")
            # print(matches_3)
            if matches_3[1] < 80:
                return None
            return matches_3[1]
        return matches_2[0]
    return matches_set[0]

def clean_text(text):
    """
    Before tokenization, normalize the text:

        - Remove extra whitespace and line breaks.
        - Convert text to lowercase (for case insensitivity).
        - Remove non-alphanumeric characters (punctuation, special symbols).
        - Normalize quotes and dashes.
    :param text: text to process
    :return: processed text
    """
    # text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces and newlines
    # text = re.sub(r'[“”‘’]', '"', text)  # Normalize quotes
    text = re.sub(r'[-—]', ' ', text)  # Normalize dashes
    # text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text

def load_data(dir_path="data", file_name="corcerers_stone_harry_poter1", file_fmt="txt"):
    file_path = f"{dir_path}/{file_name}.{file_fmt}"
    with open(file_path, "r", encoding="utf-8") as f:
        book_text = f.read()
    return book_text

book_df = pd.read_csv('data/book_df.csv')
example_chapter = book_df['processed_content'].iloc[0]
doc = nlp(example_chapter)
persons_entities = set()
persons_hist = {chapter: {person:0 for person in persons_ents} for chapter in book_df['str_idx'].to_list()}
for i,c in enumerate(book_df['processed_content'].to_list()):
    c = clean_text(c)
    doc = nlp(c)
    print("================================================================================")
    chapter = book_df['str_idx'].iloc[i]
    persons_list = [entity.text for entity in doc.ents if entity.label_ == 'PERSON']
    persons_set = set(persons_list)
    for person in list(persons_set):
        person_match = get_match(person)
        persons_entities.add(person_match)
        print(f"{person} match --> {person_match}")
        if person_match is not None:
            persons_hist[chapter][person_match] += 1
        # if person_match not in persons_hist.keys():
        #     persons_hist[chapter][person_match] = 1
        # else:
            #persons_hist[chapter][person_match] += 1

print("Final:")
print(persons_entities)
print(persons_hist)
