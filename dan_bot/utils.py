import re
import pickle
import gzip
import dill

import numpy as np
import logging
import pandas as pd


from os import makedirs
from random import random
from pathlib import Path
from collections import defaultdict
from operator import itemgetter


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from dan_bot.s3_client import S3Client


analyser = SentimentIntensityAnalyzer()

user_id = 'U0L26L3FE'


def sentiment_transform(x):
    as_dict = analyser.polarity_scores(x)
    return [as_dict['neg'], as_dict['neu'], as_dict['pos']]

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r":", "", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)


def clean_reaction(reaction):
    return re.sub(r"::skin-tone.*\d$|_\d", "", reaction)


def sub_user(text, user_dict):
    user_tags = re.findall(r"<@.*?>", text)
    if user_tags:
        for user in user_tags:
            user_name = user_dict.get(user[2:-1], 'unknown')
            text = re.sub(user, user_name, text)
    return text


def encoder_predict(enconder, text):
    if text in enconder.classes_:
        arg = np.array([text]).reshape(-1, 1)
    else:
        arg = np.array([enconder.classes_[0]]).reshape(-1, 1)
    return enconder.transform(arg)


def save_pickle(obj, file_name):
    mod_path = Path(__file__).parent
    path_to_file = (mod_path / '../input_data').resolve()
    makedirs(path_to_file, exist_ok=True)
    with open(path_to_file / file_name, 'wb') as f:
        pickle.dump(obj, f)

def load_global_model(filename):
    print('loading...', filename)
    with gzip.open(filename, 'rb') as f:
        return dill.load(f)


def load_pickle(file_name):
    try:
        mod_path = Path(__file__).parent
        path_to_file = (mod_path / '../input_data').resolve()
        makedirs(path_to_file, exist_ok=True)
        with open(path_to_file / file_name, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.debug("file: %s not found, trying from s3 bucket" % file_name)
        try:
            s3_client = S3Client()
            s3_client.resource().Bucket(
                s3_client.aws_bucket).download_file(
                'input_data/' + file_name, str(path_to_file / file_name))

            logging.debug("file: %s downloaded!" % (path_to_file / file_name))
            return load_pickle(file_name)
        except Exception as e:
            print(e)

