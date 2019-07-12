from slackclient import SlackClient
from os import getenv


dan_bot_token = getenv('DAN_BOT_KEY')
slack_client = SlackClient(dan_bot_token)
