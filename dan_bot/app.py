#!/bin/sh
import time
import logging

from pathlib import Path
from websocket import WebSocketConnectionClosedException

from utils import text_to_wordlist, sub_user, load_global_model, load_pickle
from slackbot_client import slack_client
from nn_classifier import NnClassifier

def react_with_emoji(emoji, event):
    slack_client.api_call(
        'reactions.add',
        channel=event['channel'],
        name=emoji,
        timestamp=event['ts']
    )


def connect_and_listen(global_model):
    connection = slack_client.rtm_connect()
    if connection:
        while True:
            try:
                events = slack_client.rtm_read()
                for event in events:
                    try:
                        if ('channel' in event and 'text' in event and event.get('type') == 'message'):
                            msg_text = sub_user(
                                event['text'], global_model.user_dict)
                            clean_msg = text_to_wordlist(msg_text)

                            logging.debug("Predicting for comment: %s \n channel: %s \n user: % s" % (
                                clean_msg, event['channel'], event.get('user', '')))
                            prediction = global_model.predict_emojis(
                                clean_msg, event['channel'], event.get('user', ''))
                            logging.debug("Result: %s" % prediction)

                            for emoji in prediction:
                                react_with_emoji(emoji, event)

                    except Exception as e:
                        logging.error(e)
                        logging.exception('Failed to proccess event')

                time.sleep(1)
            except WebSocketConnectionClosedException as e:
                logging.error('Caught websocket disconnect, reconnecting...')
    else:
        print('Connection failed, invalid token?')


def main():
    logging.basicConfig(level=logging.DEBUG)
    mod_path = Path(__file__).parent
    path_to_file = (mod_path / '../input_data')
    global_model = load_global_model(str(path_to_file / 'dan_bot.zip'))
    return connect_and_listen(global_model)


if __name__ == '__main__':
    main()
