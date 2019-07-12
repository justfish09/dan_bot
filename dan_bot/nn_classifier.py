import logging
from functools import wraps
from keras.models import load_model
from pathlib import Path

from utils import load_pickle
from s3_client import s3_client, aws_bucket


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
                s3_client.download_file(
                    aws_bucket, self.model_file, model_path)
                model = load_model(model_path)
            self.cache['model'] = model
        return self.cache['model']
