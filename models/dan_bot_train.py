import os
import sys
import pickle
import dill
import gzip
import pandas as pd
import numpy as np

from collections import defaultdict
from pathlib import Path
from operator import itemgetter
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras import regularizers
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score


THIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path = [THIS_DIR] + sys.path
dill.settings['recurse'] = True


from dan_bot.nn_classifier import NnClassifier
from dan_bot.s3_client import S3Client
from dan_bot.utils import sentiment_transform, load_pickle, sub_user, text_to_wordlist, clean_reaction, user_id


user_data = load_pickle('users.pkl')
channel_data = load_pickle('channel_info.pkl')

message_list = load_pickle('dan_bot_messages.pkl')

user_dict = {i['id']: i['profile']['display_name']
             for i in user_data['members']}

channel_mapping = {channel['channel']['id']: channel['channel']['name']
                   for channel in channel_data if 'channel' in channel}


def process_pkl():
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


def train_and_save(save_to_s3=True):
    # get formatted data
    long_data = process_pkl()

    # undersample popular emojis
    counts = long_data.groupby('emoji').cumcount()
    long_data = long_data[counts < 500]
    print(f'long data size: {long_data.shape[0]} rows')
    emoji_counts = long_data['emoji'].value_counts()

    # filter to most frequently used - sample. quality
    final_df = long_data[long_data['emoji'].isin(
        emoji_counts[emoji_counts > 10].index.tolist() + ['zpm', 'classic'])]

    # one hot encoding of all emojis
    formatted_table = pd.get_dummies(final_df, columns=['emoji']).groupby(
        ['comment', 'time', 'channel', 'type', 'user'], sort=False, as_index=False).max()

    # tfidf
    vectorizer = TfidfVectorizer(
        max_features=2000, stop_words='english')
    comment_vectors = vectorizer.fit_transform(
        formatted_table.comment.tolist())

    # Sentiment feature
    sentiment = np.array(
        list(formatted_table.reset_index().comment.apply(sentiment_transform).values))

    # Channel feature
    channel_encoder = LabelBinarizer()
    one_hot_channels = channel_encoder.fit_transform(
        formatted_table['channel'].values.reshape(-1, 1))

    # User feature
    user_encoder = LabelBinarizer()
    one_hot_users = user_encoder.fit_transform(
        formatted_table['user'].values.reshape(-1, 1))

    # Concat features & split train/test
    X = np.concatenate((comment_vectors.toarray(), sentiment,
                        one_hot_channels, one_hot_users), axis=1)
    y_cols = [i for i in formatted_table.columns if 'emoji' in i]
    Y = formatted_table[y_cols]

    weights = compute_class_weight(
        'balanced', final_df['emoji'].unique(), final_df['emoji'])
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, Y, test_size=0.2, random_state=1000)

    print('Shape:', Xtrain.shape, Ytrain.shape)
    print('Shape:', Xtest.shape, Ytest.shape)

    model = Sequential()
    model.add(Dense(512, activation='relu',
                    input_shape=(Xtrain.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(0.)))
    model.add(Dropout(0.5))
    model.add(Dense(Ytrain.shape[1]))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    batch_size = 64
    epochs = 35

    print(model.summary())

    early_stopping = EarlyStopping(patience=3)

    history = model.fit(
        Xtrain,
        Ytrain,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(Xtest, Ytest),
        class_weight=weights,
        callbacks=[early_stopping]
    )
    score = model.evaluate(Xtest, Ytest, batch_size=batch_size, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # print('Test F1', score[2])

    mod_path = Path(__file__).parent
    path_to_file = (mod_path / '../input_data').resolve()
    os.makedirs(path_to_file, exist_ok=True)

    keras_model_file = 'keras_model.h5'
    keras_model_path = str(path_to_file / keras_model_file)

    tf_lite_model_file = 'tf_lite_model.tflite'
    tf_lite_model_path = str(path_to_file / tf_lite_model_file)

    global_model_path = str(path_to_file / 'dan_bot.zip')

    model.save(keras_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_path)
    tflite_model = converter.convert()

    with open(tf_lite_model_path, "wb") as lite_file:
        lite_file.write(tflite_model)

    global_model = NnClassifier(vectorizer, channel_encoder, user_encoder, y_cols, keras_model_file, tf_lite_model_file, user_dict)

    with gzip.open(global_model_path, 'wb') as classifier_model_file:
        dill.dump(global_model, classifier_model_file)

    if save_to_s3:
        s3_client = S3Client()
        s3_client.client().upload_file(global_model_path, s3_client.aws_bucket, 'input_data/dan_bot.zip')
        s3_client.client().upload_file(keras_model_path, s3_client.aws_bucket, 'input_data/' + keras_model_file)
        s3_client.client().upload_file(tf_lite_model_path, s3_client.aws_bucket, 'input_data/' + tf_lite_model_file)


if __name__ == '__main__':
    train_and_save()
