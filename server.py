import time
import logging

from websocket import WebSocketConnectionClosedException

from utils import text_to_wordlist, sub_user, user_dict, channel_mapping, process_pred
from slackbot_client import slack_client


logging.basicConfig(level=logging.DEBUG)
connection = slack_client.rtm_connect()
if connection:
    while True:
        try:
            events = slack_client.rtm_read()
            for event in events:
                try:
                    if ('channel' in event and 'text' in event and event.get('type') == 'message'):
                        channel = channel_mapping.get(event['channel'], 'london')
                        text = event['text']
                        user = user_dict.get(event['user'], 'donovan.thompson')

                        msg_text = sub_user(text)
                        clean_msg = text_to_wordlist(msg_text)

                        logging.debug("Predicting for comment: %s \n channel: %s \n user: % s" % (clean_msg, channel, user))
                        prediction = process_pred(clean_msg, channel, user)
                        logging.debug("Result: %s" % prediction)

                        for emoji in prediction:
                            slack_client.api_call(
                                'reactions.add',
                                channel=event['channel'],
                                name=emoji,
                                timestamp=event['ts']
                            )
                except Exception:
                    logging.exception('Failed to proccess event')

            time.sleep(1)
        except WebSocketConnectionClosedException as e:
            logger.error('Caught websocket disconnect, reconnecting...')
else:
    print('Connection failed, invalid token?')
