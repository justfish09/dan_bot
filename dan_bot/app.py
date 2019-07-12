import time
import logging

from websocket import WebSocketConnectionClosedException

from utils import text_to_wordlist, sub_user, process_pred
from slackbot_client import slack_client
from nn_classifier import NnClassifier


def react_with_emoji(emoji, event):
    slack_client.api_call(
        'reactions.add',
        channel=event['channel'],
        name=emoji,
        timestamp=event['ts']
    )


def connect_and_listen(model_class):
    connection = slack_client.rtm_connect()
    if connection:
        while True:
            try:
                events = slack_client.rtm_read()
                for event in events:
                    try:
                        if ('channel' in event and 'text' in event and event.get('type') == 'message'):
                            channel = model_class.channel_mapping().get(event['channel'], 'london')
                            user = model_class.user_dict().get(event['user'], 'donovan.thompson')

                            msg_text = sub_user(event['text'], model_class.user_dict())
                            clean_msg = text_to_wordlist(msg_text)

                            logging.debug("Predicting for comment: %s \n channel: %s \n user: % s" % (clean_msg, channel, user))
                            prediction = process_pred(clean_msg, channel, user, model_class)
                            logging.debug("Result: %s" % prediction)

                            for emoji in prediction:
                                react_with_emoji(emoji, event)

                    except Exception as e:
                        logging.error(e)
                        logging.exception('Failed to proccess event')

                time.sleep(2)
            except WebSocketConnectionClosedException as e:
                logging.error('Caught websocket disconnect, reconnecting...')
    else:
        print('Connection failed, invalid token?')


def main():
    logging.basicConfig(level=logging.DEBUG)
    model_class = NnClassifier()
    # model = model_class.load_classifier()
    return connect_and_listen(model_class)

if __name__ == '__main__':
    main()
