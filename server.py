from slackclient import SlackClient
import time
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from keras.models import load_model
import keras.metrics
import numpy as np
from os import getenv
import pickle
import logging


from utils import f1, text_to_wordlist
from utils import clean_reaction, sub_user, encoder_predict, user_dict, channel_mapping


keras.metrics.f1 = f1

dan_bot_token = getenv('DAN_BOT_KEY')
print('token', dan_bot_token)
slack_client = SlackClient(dan_bot_token)

# model and vectorizers
model = load_model('my_model.h5')
vectorizer = pickle.load(open("input_data/tfidf.pickle", "rb"))
channel_encoder = pickle.load(open("input_data/channel_enc.pickle", "rb"))
user_encoder = pickle.load(open("input_data/user_enc.pickle", "rb"))
y_cols = pickle.load(open("input_data/y_cols.pickle", "rb"))

analyser = SentimentIntensityAnalyzer()


def process_pred(sentence, channel, user):
    xpred = vectorizer.transform([sentence])
    new_x = np.concatenate(
        (
            xpred.toarray(),
            np.array([analyser.polarity_scores(sentence)
                      ['compound']]).reshape(-1, 1),
            encoder_predict(channel_encoder, channel).toarray(),
            encoder_predict(user_encoder, user).toarray()
        ), axis=1)
    pred = model.predict(new_x)
    return y_cols[np.argmax(pred)].split('emoji_')[-1]


logging.basicConfig(level=logging.DEBUG)
connection = slack_client.rtm_connect()
if connection:
    logging.debug("the connection is %s" % connection)
    while True:
        events = slack_client.rtm_read()
        logging.debug("events are: %s" % events)
        for event in events:
            if ('channel' in event and 'text' in event and event.get('type') == 'message'):
                channel = channel_mapping.get(event['channel'], 'london')
                text = event['text']
                user = user_dict.get(event['user'], 'donovan.thompson')

                msg_text = sub_user(text)
                clean_msg = text_to_wordlist(msg_text)

                logging.debug(" Prepdicting for comment: %s \n channel: %s \n user: % s" % (clean_msg, channel, user))
                prediction = process_pred(clean_msg, channel, user)
                logging.debug(" Result: %s" % prediction)

                post = slack_client.api_call(
                    'reactions.add',
                    channel=event['channel'],
                    name=prediction,
                    timestamp=event['ts']
                )

        time.sleep(1)
else:
    print('Connection failed, invalid token?')
