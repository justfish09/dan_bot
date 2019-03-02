import re
import pickle
import numpy as np
import logging

import keras.metrics

from os import getenv
from string import punctuation

from slackclient import SlackClient
import boto3
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


def sentiment_transform(x):
    as_dict = analyser.polarity_scores(x)
    return [as_dict['neg'], as_dict['neu'], as_dict['pos']]


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
    return [col_name.split('emoji_')[-1] for col_name, score in zip(y_cols, pred) if score > 0.05]


def process_pred(sentence, channel, user):
    return process_pred_specified_models(
        sentence,
        channel,
        user,
        vectorizer,
        channel_encoder,
        user_encoder,
        y_cols
    )


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


def sub_user(text):
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


def load_pickle(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.debug("file: %s not found, trying from s3 bucket" % file_name)
        try:
            s3_client.download_file(aws_bucket, file_name, file_name)
            logging.debug("file: %s downloaded!" % file_name)
            return load_pickle(file_name)
        except Exception as e:
            print(e)


aws_id = getenv('DAN_BOT_AWS_ID')
aws_key = getenv('DAN_BOT_AWS_KEY')
aws_bucket = getenv('DAN_BOT_BUCKET')

slack_token = getenv('MY_SLACK_KEY')

s3_client = boto3.client('s3', aws_access_key_id=aws_id,
                         aws_secret_access_key=aws_key)

vectorizer = load_pickle('input_data/tfidf.pickle')
channel_encoder = load_pickle("input_data/channel_enc.pickle")
user_encoder = load_pickle("input_data/user_enc.pickle")
y_cols = load_pickle("input_data/y_cols.pickle")
user_list = load_pickle('input_data/users.pkl')
channel_info = load_pickle('input_data/channel_info.pkl')

keras.metrics.f1 = f1
model_file = 'my_model.h5'

try:
    model = load_model(model_file)
except Exception as e:
    s3_client.download_file(aws_bucket, model_file, model_file)
    model = load_model(model_file)

analyser = SentimentIntensityAnalyzer()

user_id = 'U0L26L3FE'

user_dict = {i['id']: i['profile']['display_name']
             for i in user_list['members']}

channel_mapping = {channel['channel']['id']: channel['channel']['name']
                   for channel in channel_info if 'channel' in channel}
