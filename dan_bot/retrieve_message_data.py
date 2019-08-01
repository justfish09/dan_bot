from operator import itemgetter
import pickle
from utils import user_id
from slackbot_client import slack_client
import time
from pathlib import Path


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


def main():
    user_msgs = retrieve_reaction_data(user_id)
    other_msgs = retrieve_reaction_data('U144M1H4Z')
    all_msgs = [*user_msgs, *other_msgs]

    dir_path = Path(__file__)
    input_path = (dir_path / '../../input_data/dan_bot_messages.pkl').resolve()
    with open(input_path, 'wb') as f:
        pickle.dump(all_msgs, f)


if __name__ == '__main__':
    main()
