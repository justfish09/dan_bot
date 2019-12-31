import os
import sys
import pickle
import time

from pathlib import Path
from operator import itemgetter

from dan_bot.utils import user_id
from dan_bot.slackbot_client import slack_client

THIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path = [THIS_DIR] + sys.path


def retrieve_reaction_data(user_id):
    current_page = 1
    start = 0
    first_call = slack_client.api_call(
        "reactions.list", user=user_id, page=current_page)
    if not first_call['ok']:
        print('Could not connect to reaction list, retrying', start)
        if start <= 3:
            retrieve_reaction_data(user_id)
        else:
            quit()

    message_info = first_call['items']
    current_page, total_pages = itemgetter(
        'page', 'pages')(first_call['paging'])
    print('total_pages: ', total_pages)
    while (current_page) <= total_pages:
        try:
            time.sleep(1)
            print(current_page, first_call['ok'], len(message_info))
            first_call = slack_client.api_call(
                "reactions.list", user=user_id, page=current_page)
            message_info = [*message_info, *first_call['items']]
            current_page += 1
        except Exception as e:
            print('failed call on page %d with error: %s ', (current_page, e))
            continue
    return message_info


def save_users():
    users_response = slack_client.api_call('users.list')
    data = users_response['members']
    with open(f'{THIS_DIR}/input_data/users.pkl', 'wb') as f:
        pickle.dump(data, f)


def save_channels():
    channels_response = slack_client.api_call('channels.list')
    data = channels_response['channels']
    with open(f'{THIS_DIR}/input_data/channel_info.pkl', 'wb') as f:
        pickle.dump(data, f)


def save_reactions():
    user_msgs = retrieve_reaction_data(user_id)
    other_msgs = retrieve_reaction_data('U144M1H4Z')
    mikes_msgs = retrieve_reaction_data('UC0EEMCPN')
    all_msgs = [*user_msgs, *other_msgs, *mikes_msgs]
    with open(f'{THIS_DIR}/input_data/dan_bot_messages.pkl', 'wb') as f:
        pickle.dump(all_msgs, f)


def main():
    print('saving users')
    # save_users()
    print('saving channels')
    # save_channels()
    print('saving reactions')
    save_reactions()


if __name__ == '__main__':
    main()
