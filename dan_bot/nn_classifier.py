import logging
import platform

import pandas as pd
import numpy as np

from functools import wraps
from pathlib import Path
from random import random

from dan_bot.utils import sentiment_transform, encoder_predict
from dan_bot.s3_client import S3Client


class NnClassifier(object):
    """A class that contains the neural network classifier"""

    def __init__(self, comment_vectorizer, channel_vectorizer,
                 user_vectorizer, emoji_labels, keras_model_file,
                 lite_model_file, user_dict):
        self.comment_vectorizer = comment_vectorizer
        self.channel_vectorizer = channel_vectorizer
        self.user_vectorizer = user_vectorizer
        self.emoji_labels = emoji_labels
        self.keras_model_file = keras_model_file
        self.lite_model_file = lite_model_file
        self.user_dict = user_dict
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

    def predict_emojis(self, sentence, channel, user):
        return [emoji for emoji, score in self.process_pred_specified_models(
            sentence,
            channel,
            user
        ) if score > random()]

    def create_features(self, sentence, channel, user):
        xpred = self.comment_vectorizer.transform([sentence])
        return np.concatenate(
            (
                xpred.toarray(),
                np.array([sentiment_transform(sentence)]),
                encoder_predict(self.channel_vectorizer, channel),
                encoder_predict(self.user_vectorizer, user)
            ), axis=1).astype(np.float32)

    def predict_tf_lite(self, sentence, channel, user):
        new_x = self.create_features(sentence, channel, user)

        if platform.machine() == 'aarch64':         # raspberry pi - use tensorflowlite
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path='../input_data/'+self.lite_model_file)
        else:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path='../input_data/'+self.lite_model_file)

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        interpreter.set_tensor(input_details[0]['index'], new_x)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]
        return sorted([
            (col_name.split('emoji_')[-1], score)
            for col_name, score in zip(self.emoji_labels, pred)
            if score > 0.05
        ], key=lambda x: x[1])

    def process_pred_specified_models(self, sentence,
                                      channel, user):
        new_x = self.create_features(sentence, channel, user)
        if self.model is None:
            self.model = self.load_classifier()

        if platform.machine() == 'aarch64':
            return self.predict_tf_lite(sentence, channel, user)

        else:
            pred = self.model.predict(new_x)[0]
        return sorted([
            (col_name.split('emoji_')[-1], score)
            for col_name, score in zip(self.emoji_labels, pred)
            if score > 0.05
        ], key=lambda x: x[1])

