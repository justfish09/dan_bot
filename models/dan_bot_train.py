import os
import sys
import pickle
import dill
import gzip
import pandas as pd
import numpy as np
import json

from collections import defaultdict
from pathlib import Path
from operator import itemgetter
from typing import Dict, Callable
import tensorflow as tf

from keras.models import Sequential
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Activation, multiply
from keras.models import Model
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

THIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path = [THIS_DIR] + sys.path
dill.settings['recurse'] = True


from dan_bot.nn_classifier import NnClassifier
from dan_bot.s3_client import S3Client
from dan_bot.utils import SentimentVectorizer, load_pickle, sub_user, text_to_wordlist, clean_reaction, user_id


user_data = load_pickle('users.pkl')
channel_data = load_pickle('channel_info.pkl')

message_list = load_pickle('dan_bot_messages.pkl')

user_dict = {i['id']: i['profile']['display_name']
             for i in user_data}

channel_mapping = {channel['channel']['id']: channel['channel']['name']
                   for channel in channel_data if 'channel' in channel}

keras_model_file = 'keras_model.h5'
tf_lite_model_file = 'tf_lite_model.tflite'


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
                        # user_id in reaction['users'] or 'U144M1H4Z' in reaction['users']
                        True
                        )
                    ]
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


def vectorizer_store(tfidf_size_comment):
    return {
        'comment': {'is_binary': False, 'vectorizer': TfidfVectorizer(max_features=tfidf_size_comment, stop_words='english')},
        'sentiment': {'is_binary': True, 'vectorizer': SentimentVectorizer(), 'key_override': 'comment'},
        'channel': {'is_binary': True, 'vectorizer': LabelBinarizer()},
        'user': {'is_binary': True, 'vectorizer': LabelBinarizer()},
    }


def vec_transform(data_table, vectorizer_store, key):
    data_key = vectorizer_store[key].get('key_override', key)
    data = data_table[data_key]
    if data.dtype == 'O':
        data = data.fillna('na').str.lower()
    else:
        data = data.fillna(0)

    transformed = vectorizer_store[key]['vectorizer'].transform(data)
    if vectorizer_store[key]['is_binary']:
        return transformed
    return transformed.toarray()


def fit_all(data_table, vectorizer_store):
    for key in vectorizer_store:
        data_key = vectorizer_store[key].get('key_override', key)
        data = data_table[data_key]
        if data.dtype == 'O':
            data = data.fillna('na').str.lower()
        else:
            data = data.fillna(0)

        vectorizer_store[key]['vectorizer'].fit(data)
    return


def create_features(data: pd.DataFrame,
                    vectorizer_store: Dict,
                    columns: list) -> pd.DataFrame:
    if len(columns) == 1:
        return vec_transform(data, vectorizer_store, columns[0])
    return np.concatenate(
        [
            vec_transform(
                data,
                vectorizer_store,
                col
            ) for col in columns
        ],
        axis=1
    )


def save_metrics(filename: str, metrics: Dict) -> None:
    with open(filename, 'w') as metrics_file:
        json.dump({'metrics': metrics}, metrics_file, indent=4)


def optimize(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    space = {
        'hidden_units_1': hp.uniform('hidden_units_1', 64, 256),
        'hidden_units_2': hp.uniform('hidden_units_2', 28, 256),
        'hidden_categ_units_1': hp.uniform('hidden_categ_units_1', 64, 256),
        'dropout1': hp.uniform('dropout1', .25, .5),
        'dropout2': hp.uniform('dropout2',  .25, .5),
        'dropout_categ': hp.uniform('dropout_categ',  .1, .4),
        'batch_size': hp.uniform('batch_size', 28, 128),
        'tfidf_feats': hp.uniform('tfidf_feats', 50, 500),
        'nb_epochs':  hp.uniform('nb_epochs', 30, 100),
        'threshold': hp.uniform('threshold',  .35, .5),
        'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
        'activation': hp.choice('activation', ['relu', 'tanh', 'exponential']),
    }
    trials = Trials()
    best = fmin(evaluate_loss(train, test, keras_model_file, tf_lite_model_file),
                space, algo=tpe.suggest, max_evals=20, trials=trials)
    best = space_eval(space, best)
    return best, trials


def evaluate_model(model: Model, xtest: np.array,
                   ytest: np.array, threshold: float) -> Dict:
    loss, accuracy = model.evaluate(xtest, ytest)
    pred_test = (model.predict(xtest) >= threshold).astype(int)
    f1 = f1_score(ytest, pred_test, average='micro')
    return {'loss': loss, 'accuracy': accuracy, 'micro_f1': f1}


def evaluate_loss(train_set: pd.DataFrame, test_set: pd.DataFrame,
                  keras_model_file: str, tf_lite_model_file: str) -> Callable:
    def create_model(params: Dict, return_model=False) -> Dict:
        print('testing params: ', params)

        y_cols = [i for i in train_set.columns if 'emoji_' in i]
        Ytrain = train_set[y_cols].values
        Ytest = test_set[y_cols].values

        all_data = pd.concat((train_set, test_set))

        # weights = compute_class_weight(
        #     'balanced', all_data['emoji'].unique(), all_data['emoji'])

        vec_store = vectorizer_store(
            int(params['tfidf_feats'])
        )

        # fit all vectorizers
        fit_all(all_data, vec_store)

        # create comment tfidf
        tfidf_xtrain = create_features(train_set, vec_store, ['comment'])
        tfidf_xtest = create_features(test_set, vec_store, ['comment'])

        # create categorical feats
        categ_cols = [i for i in vec_store if i != 'comment']
        categ_xtrain = create_features(train_set, vec_store, categ_cols)
        categ_xtest = create_features(test_set, vec_store, categ_cols)
        print(categ_xtrain)

        # Comment TFIDF Network
        tfdif_input = Input(shape=(tfidf_xtrain.shape[1],),
                            name='tfidf_input')
        tfdif_dense_1 = Dense(int(params['hidden_units_1']),
                              activation=params['activation'],
                              name='tfdif_dense_1')(tfdif_input)
        tfdif_dropout_1 = Dropout(params['dropout1'],
                                  name='tfdif_dropout_1')(tfdif_dense_1)
        tfdif_dense_2 = Dense(int(params['hidden_units_2']),
                              activation=params['activation'],
                              name='tfdif_dense_2')(tfdif_dropout_1)

        # Categorical Network
        categ_input = Input(shape=(categ_xtrain.shape[1], ),
                            name='categ_input')
        categ_dense_1 = Dense(int(params['hidden_categ_units_1']),
                              activation=params['activation'],
                              name='categ_dense_1')(categ_input)
        categ_dropout_1 = Dropout(params['dropout_categ'],
                                  name='categ_dropout_1')(categ_dense_1)
        categ_dense_2 = Dense(int(params['hidden_units_2']),
                              activation=params['activation'],
                              name='categ_dense_2')(categ_dropout_1)

        # Merge outputs of both networks multiply strategy
        merged = multiply([tfdif_dense_2, categ_dense_2])

        # add one more dense & droput layer before output
        post_merge = Dense(Ytrain.shape[1], activation='relu', name='post_merge')(merged)
        last_drop = Dropout(params['dropout2'], name='B6')(post_merge)

        # output layer
        output = Dense(Ytrain.shape[1], activation='sigmoid', name='output')(last_drop)

        merged_model = Model(inputs=[tfdif_input, categ_input], outputs=[output])
        merged_model.compile(loss='categorical_crossentropy',
                             optimizer='adam', metrics=['accuracy'])

        print(merged_model.summary())

        merged_model.fit(
            [tfidf_xtrain, categ_xtrain],
            Ytrain,
            batch_size=int(params['batch_size']),
            epochs=int(params['nb_epochs']),
            verbose=2,
            validation_data=([tfidf_xtest, categ_xtest], Ytest),
            callbacks=[EarlyStopping(patience=5)],
            # class_weight=weights
        )

        model_scores = evaluate_model(merged_model, [tfidf_xtest, categ_xtest], Ytest, params['threshold'])
        print('Metrics:', model_scores)

        if return_model:
            plot_model(merged_model, to_file='dan_bot.png')
            return (
                merged_model,
                NnClassifier(
                    vec_store,
                    y_cols,
                    keras_model_file,
                    tf_lite_model_file,
                    user_dict,
                    categ_cols
                ),
                model_scores
            )

        return {'loss': 1.0 - model_scores['micro_f1'], 'status': STATUS_OK}

    return create_model


def save_classifier(keras_model: Model, global_model: NnClassifier,
                    save_to_s3=True) -> None:
    mod_path = Path(__file__).parent
    path_to_file = (mod_path / '../input_data').resolve()
    os.makedirs(path_to_file, exist_ok=True)
    keras_model_path = str(path_to_file / keras_model_file)
    tf_lite_model_path = str(path_to_file / tf_lite_model_file)
    global_model_path = str(path_to_file / 'dan_bot.zip')

    keras_model.save(keras_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_path)
    tflite_model = converter.convert()

    with open(tf_lite_model_path, "wb") as lite_file:
        lite_file.write(tflite_model)

    with gzip.open(global_model_path, 'wb') as classifier_model_file:
        dill.dump(global_model, classifier_model_file)

    if save_to_s3:
        s3_client = S3Client()
        s3_client.client().upload_file(global_model_path, s3_client.aws_bucket,
                                       'input_data/dan_bot.zip')
        s3_client.client().upload_file(keras_model_path, s3_client.aws_bucket,
                                       'input_data/' + keras_model_file)
        s3_client.client().upload_file(tf_lite_model_path, s3_client.aws_bucket,
                                       'input_data/' + tf_lite_model_file)


def main():
    model_name = 'dan_bot'
    model_path = f'{THIS_DIR}/models/{model_name}'

    long_data = process_pkl()

    # undersample popular emojis
    # TODO change to imbalance learn
    counts = long_data.groupby('emoji').cumcount()
    long_data = long_data[counts < 500]
    print(f'long data size: {long_data.shape[0]} rows')
    emoji_counts = long_data['emoji'].value_counts()

    # filter to most frequently used - sample. quality
    final_df = long_data[long_data['emoji'].isin(
        emoji_counts[emoji_counts > 15].index.tolist() + ['zpm', 'classic', 'elephant', 'richard', 'happy_steve', 'sandbox', 'pathetic'])]

    # one hot encoding of all emojis
    formatted_table = pd.get_dummies(final_df, columns=['emoji']).groupby(
        ['comment', 'time', 'channel', 'type', 'user'], sort=False, as_index=False).max()

    train_set, test_set = train_test_split(
        formatted_table, test_size=0.2, random_state=1000)

    print('train_set shape: ', train_set.shape)
    print('test_set shape: ', test_set.shape)
    print(formatted_table.head())

    print('hyperopting')
    best_params, trials = optimize(train_set, test_set)
    print('\nbest_params', best_params)

    keras_model, classifier, scores = evaluate_loss(
        train_set, test_set, keras_model_file, tf_lite_model_file
    )(best_params, return_model=True)

    print('saving models & artefacts')
    save_classifier(keras_model, classifier)
    save_metrics(f'{model_path}_metrics.json', scores)

    with open(f'{model_path}_trials.pickle', 'wb') as f:
        dill.dump(trials, f)


if __name__ == '__main__':
    main()
