import re
import pickle
import numpy as np

import keras.metrics

from os import getenv
from string import punctuation
from random import random

from slackclient import SlackClient
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras import backend as K
from keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


keras.metrics.f1 = f1
model = load_model('my_model.h5')

analyser = SentimentIntensityAnalyzer()


def sentiment_transform(x):
    as_dict = analyser.polarity_scores(x)
    return [as_dict['neg'], as_dict['neu'], as_dict['pos']]


with open('input_data/tfidf.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

with open("input_data/channel_enc.pickle", "rb")as f:
    channel_encoder = pickle.load(f)

with open("input_data/user_enc.pickle", "rb") as f:
    user_encoder = pickle.load(f)

with open("input_data/y_cols.pickle", "rb")as f:
    y_cols = pickle.load(f)


def process_pred_specified_models(
    sentence,
    channel,
    user,
    vectorizer,
    channel_encoder,
    user_encoder,
    y_cols):
    xpred = vectorizer.transform([sentence])
    new_x = np.concatenate(
        (
            xpred.toarray(),
            np.array([sentiment_transform(sentence)]),
            encoder_predict(channel_encoder, channel),
            encoder_predict(user_encoder, user)
        ), axis=1)
    pred = model.predict(new_x)[0]
    return sorted([
            (col_name.split('emoji_')[-1], score)
            for col_name, score in zip(y_cols, pred)
            if score > 0.05
    ], key=lambda x: x[1])


def process_pred(sentence, channel, user):
    return [emoji for emoji, score in process_pred_specified_models(
        sentence,
        channel,
        user,
        vectorizer,
        channel_encoder,
        user_encoder,
        y_cols
    ) if score > random()]


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
    return re.sub(r"::skin-tone.*\d$", "", reaction)


def sub_user(text):
    user_tags = re.findall(r"<@.*?>", text)
    if user_tags:
        for user in user_tags:
            user_name = user_dict.get(user[2:-1], 'unknown')
            text = re.sub(user, user_name, text)
    return text


def encoder_predict(enconder, text):
    arg = np.array([text]).reshape(-1, 1)
    return enconder.transform(arg)


# user list
with open('input_data/users.pkl', 'rb') as f:
    user_list = pickle.load(f)

# channel list
with open('input_data/channel_info.pkl', 'rb') as f:
    channel_info = pickle.load(f)

slack_token = getenv('MY_SLACK_KEY')
# sc = SlackClient(slack_token)
user_id = 'U0L26L3FE'

user_dict = {i['id']: i['profile']['display_name']
             for i in user_list['members']}
channel_mapping = {channel['channel']['id']: channel['channel']['name']
                   for channel in channel_info if 'channel' in channel}
