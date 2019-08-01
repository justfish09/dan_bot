import logging
from functools import wraps
from keras.models import load_model
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import EarlyStopping

from utils import load_pickle, process_pkl, f1, sentiment_transform, process_pred_specified_models, save_pickle
from s3_client import S3Client


class NnClassifier(object):
    """A class that contains the neural network classifier"""

    def __init__(self):
        self.vectorizer_path = 'tfidf.pickle'
        self.channel_encoder_path = "channel_enc.pickle"
        self.user_encoder_path = "user_enc.pickle"
        self.emoji_labels_path = "y_cols.pickle"
        self.user_list_path = 'users.pkl'
        self.channel_info_path = 'channel_info.pkl'
        self.model_file = '../my_model.h5'
        self.cache = {}
        self.tfidf_max_words = 5000
        self.persist_data = True

    def cache_return(self, file):
        if file not in self.cache:
            self.cache[file] = load_pickle(file)
        return self.cache[file]

    def vectorizer(self):
        return self.cache_return(self.vectorizer_path)

    def channel_encoder(self):
        return self.cache_return(self.channel_encoder_path)

    def emoji_labels(self):
        return self.cache_return(self.emoji_labels_path)

    def user_encoder(self):
        return self.cache_return(self.user_encoder_path)

    def user_dict(self):
        user_list = self.cache_return(self.user_list_path)
        return {i['id']: i['profile']['display_name']
                for i in user_list['members']}

    def channel_mapping(self):
        channel_info = self.cache_return(self.channel_info_path)
        return {channel['channel']['id']: channel['channel']['name']
                for channel in channel_info if 'channel' in channel}

    def load_classifier(self):
        mod_path = Path(__file__).parent
        model_path = (mod_path / self.model_file).resolve()
        if 'model' not in self.cache:
            try:
                model = load_model(str(model_path))
            except Exception as e:
                logging.info('failed to load model file, checking s3...')
                s3_client = S3Client()
                s3_client.resource().Bucket(s3_client.aws_bucket).download_file('my_model.h5', str(model_path))
                model = load_model(str(model_path))
            self.cache['model'] = model
        return self.cache['model']

    def train(self):
        # get formatted data
        long_data = process_pkl(self.channel_mapping(), self.user_dict())

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
        tokenizer = Tokenizer(num_words=self.tfidf_max_words)
        vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_words, stop_words='english')
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
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy', f1])
        print(model.metrics_names)

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
            # callbacks=[early_stopping]
        )
        score = model.evaluate(Xtest, Ytest, batch_size=batch_size, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('Test F1', score[2])

        for sentence in self.test_senteces():
            pred = process_pred_specified_models(
                sentence,
                'phat_data_public',
                'richardf',
                model,
                vectorizer,
                channel_encoder,
                user_encoder,
                y_cols
            )
            print(sentence, pred)

        if self.persist_data:
            save_pickle(vectorizer, self.vectorizer_path)
            save_pickle(channel_encoder, self.channel_encoder_path)
            save_pickle(user_encoder, self.user_encoder_path)
            save_pickle(y_cols, self.emoji_labels_path)
            model.save(self.model_file)

            # for file in file_store:
            mod_path = Path(__file__).parent
            path_to_file = (mod_path / '../input_data')
            s3_client = S3Client()
            s3_client.client().upload_file(str((path_to_file / self.vectorizer_path).resolve()), s3_client.aws_bucket, 'input_data/' + self.vectorizer_path)
            s3_client.client().upload_file(str((path_to_file / self.channel_encoder_path).resolve()), s3_client.aws_bucket, 'input_data/' + self.channel_encoder_path)
            s3_client.client().upload_file(str((path_to_file / self.user_encoder_path).resolve()), s3_client.aws_bucket, 'input_data/' + self.user_encoder_path)
            s3_client.client().upload_file(str((path_to_file / self.emoji_labels_path).resolve()), s3_client.aws_bucket, 'input_data/' + self.emoji_labels_path)
            s3_client.client().upload_file(str((mod_path / self.model_file).resolve()), s3_client.aws_bucket, 'my_model.h5')


    def test_senteces(self):
        return [
            'brexit makes me sad',
            'great job on getting autocoding out, you massive nerds',
            'on leave that week',
            'production is down',
            'just be better',
            'work harder',
            'rocket to production',
            'thats just wrong',
            'windows over mac',
            'you are a bell end',
            "it's not unreasonable to have a w9am meeting",
            "My understanding from talking to different folks is the issue is due to the different text length",
            '@steven.perianen IBM is loving the new verbatim auto coding!',
            'heyhey @daniel.baark -> https://zigroup.atlassian.net/browse/SP-5320',
            "The new DS review time clashes with another meeting",
            "It's not like me to skip meals",
            "There has been a complaint about people using the putney office and keeping the door propped open. Can people make sure the door isn't kept open when it shouldn't be.",
            "Ahh we call them a Microsoft Product Team"
        ]
