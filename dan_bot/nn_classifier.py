import logging
import platform

import pandas as pd
import numpy as np

from functools import wraps
from pathlib import Path
from random import random

from dan_bot.utils import SentimentVectorizer, encoder_predict
from dan_bot.s3_client import S3Client


class NnClassifier(object):
    """A class that contains the neural network classifier"""

    def __init__(self, vectorizer_store, emoji_labels, keras_model_file,
                 lite_model_file, user_dict, categ_cols):
        self.vectorizer_store = vectorizer_store
        self.emoji_labels = emoji_labels
        self.keras_model_file = keras_model_file
        self.lite_model_file = lite_model_file
        self.user_dict = user_dict
        self.categ_cols = categ_cols
        self.model = None

    def load_classifier(self):
        mod_path = Path(__file__).parent
        path_to_file = (mod_path / '../').resolve()

        if platform.machine() == 'aarch64':         # raspberry pi - use tensorflowlite
            import tflite_runtime.interpreter as tflite
            model = tflite.Interpreter(model_path=str(path_to_file / 'tflite_model' / self.lite_model_file))
        else:
            from keras.models import load_model
            try:
                model = load_model(str(path_to_file / 'keras_model' /  self.keras_model_file))
            except Exception as e:
                logging.info('failed to load model file, checking s3...')
                s3_client = S3Client()
                s3_client.resource().Bucket(s3_client.aws_bucket).download_file('my_model.h5', str(model_path))
                model = load_model(str(model_path))
        return model

    def create_categ_input(self, features) -> np.array:
        data_store = []
        for col in self.categ_cols:
            data_key = self.vectorizer_store[col].get('key_override', col)
            transformed = self.vectorizer_store[col]['vectorizer'].transform(features[data_key])
            if not self.vectorizer_store[col]['is_binary']:
                transformed = transformed.toarray()
            data_store.append(transformed)
        return np.concatenate(data_store, axis=1)

    def create_comment_tfidf(self, features: dict) -> np.array:
        return self.vectorizer_store['comment']['vectorizer'].transform(
          features['comment']).toarray()


    def predict_emojis(self, features):
        return [emoji for emoji, score in self.process_pred_specified_models(
            features
        ) if score > random()]

    def create_features(self, features):
        x_tfidf = self.create_comment_tfidf(features).astype(np.float32)
        x_categ = self.create_categ_input(features).astype(np.float32)
        return [x_tfidf, x_categ]

    def predict_tf_lite(self, features):
        mod_path = Path(__file__).parent
        path_to_file = (mod_path / '../input_data')
        new_x = self.create_features(features)

        if platform.machine() == 'aarch64':         # raspberry pi - use tensorflowlite
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(str(path_to_file / self.lite_model_file))
        else:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(str(path_to_file / self.lite_model_file))

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], new_x[0])
        interpreter.set_tensor(input_details[1]['index'], new_x[1])
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]
        return sorted([
            (col_name.split('emoji_')[-1], score)
            for col_name, score in zip(self.emoji_labels, pred)
            if score > 0.5
        ], key=lambda x: x[1])

    def process_pred_specified_models(self, features):
        new_x = self.create_features(features)
        if self.model is None:
            self.model = self.load_classifier()

        if platform.machine() == 'aarch64':
            return self.predict_tf_lite(features)

        else:
            pred = self.model.predict(new_x)[0]
        return sorted([
            (col_name.split('emoji_')[-1], score)
            for col_name, score in zip(self.emoji_labels, pred)
            if score > 0.5
        ], key=lambda x: x[1])

