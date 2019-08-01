import re
import pickle
import numpy as np
import logging
import keras.metrics
import pandas as pd


from os import makedirs
from random import random
from pathlib import Path
from collections import defaultdict
from operator import itemgetter


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras import backend as K
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from s3_client import S3Client


analyser = SentimentIntensityAnalyzer()

user_id = 'U0L26L3FE'


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


def sentiment_transform(x):
    as_dict = analyser.polarity_scores(x)
    return [as_dict['neg'], as_dict['neu'], as_dict['pos']]


def process_pred_specified_models(
        sentence,
        channel,
        user,
        model,
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


def process_pred(sentence, channel, user, model_class):
    return [emoji for emoji, score in process_pred_specified_models(
        sentence,
        channel,
        user,
        model_class.load_classifier(),
        model_class.vectorizer(),
        model_class.channel_encoder(),
        model_class.user_encoder(),
        model_class.emoji_labels()
    ) if score > random() / 1.5]


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


def process_pkl(channel_mapping, user_dict):
    mod_path = Path(__file__).parent
    path_to_file = (mod_path / '../input_data/dan_bot_messages.pkl').resolve()
    with open(path_to_file, 'rb') as f:
        message_list = pickle.load(f)
        print('number of msgs loaded: ', len(message_list))

    store = {}
    emoji_count = defaultdict(int)

    for msg in message_list:
        try:
            if 'message' in msg:
                msg_type, msg_info, channel = itemgetter(
                    'type', 'message', 'channel')(msg)
                msg_info_type, msg_text, msg_reactions, msg_time = itemgetter(
                    'type', 'text', 'reactions', 'ts')(msg['message'])
                if 'message' in msg and msg_text.strip != '':
                    msg_text = sub_user(msg_text, user_dict)
                    clean_msg = text_to_wordlist(msg_text)
                    store[clean_msg] = {}
                    store[clean_msg]['reactions'] = [clean_reaction(reaction['name']) for reaction in msg_reactions if (
                        user_id in reaction['users'] or 'U144M1H4Z' in reaction['users'])]
                    store[clean_msg]['time'] = msg_time
                    store[clean_msg]['type'] = 'message'
                    store[clean_msg]['joined_reactions'] = '|'.join(
                        store[clean_msg]['reactions'])
                    store[clean_msg]['channel'] = channel
                    store[clean_msg]['type'] = msg['type']
                    if 'user' in msg['message']:
                        store[clean_msg]['user'] = msg['message']['user']
                    elif 'bot_id' in msg['message']:
                        store[clean_msg]['user'] = msg['message']['bot_id']
                    for emoji in store[clean_msg]:
                        emoji_count[emoji] += 1
            elif 'file' in msg:
                continue
        except Exception as e:
            print(e)

    long_store = []
    for k, v in store.items():
        for reaction in v['reactions']:
            long_store.append(
                {'comment': k,
                 'emoji': reaction,
                 'channel': channel_mapping.get(v['channel'], 'private'),
                 'time': float(v['time']),
                 'user': user_dict.get(v.get('user', 'None'), 'bot'),
                 'type': v['type']}
            )

    long_data = pd.DataFrame(long_store)
    return long_data
