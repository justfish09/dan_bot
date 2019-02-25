from slackclient import SlackClient
import time
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import numpy as np
from os import getenv
import pickle
import logging

from websocket import WebSocketConnectionClosedException


from utils import f1, text_to_wordlist, clean_reaction, sub_user, encoder_predict, user_dict, channel_mapping, process_pred

dan_bot_token = getenv('DAN_BOT_KEY')
slack_client = SlackClient(dan_bot_token)

logging.basicConfig(level=logging.DEBUG)
connection = slack_client.rtm_connect()
if connection:
    logging.debug("the connection is %s" % connection)
    while True:
        try:
            events = slack_client.rtm_read()
            logging.debug("events are: %s" % events)
            for event in events:
                if ('channel' in event and 'text' in event and event.get('type') == 'message'):
                    channel = channel_mapping.get(event['channel'], 'london')
                    text = event['text']
                    user = user_dict.get(event['user'], 'donovan.thompson')

                    msg_text = sub_user(text)
                    clean_msg = text_to_wordlist(msg_text)

                    logging.debug(" Predicting for comment: %s \n channel: %s \n user: % s" % (clean_msg, channel, user))
                    prediction = process_pred(clean_msg, channel, user)
                    logging.debug(" Result: %s" % prediction)

                    post = slack_client.api_call(
                        'reactions.add',
                        channel=event['channel'],
                        name=prediction,
                        timestamp=event['ts']
                    )

            time.sleep(1)
        except WebSocketConnectionClosedException as e:
            logger.error('Caught websocket disconnect, reconnecting...')
            # slack_client.rtm_connect(auto_reconnect=True)
else:
    print('Connection failed, invalid token?')
