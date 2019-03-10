from operator import itemgetter
import pickle
from utils import user_id
from slack_client import slack_client

def retrieve_reaction_data(user_id):
    current_page = 1
    first_call = slack_client.api_call(
        "reactions.list", user=user_id, page=current_page)
    if not first_call['ok']:
        raise Exception('Could not connect to reaction list')

    message_info = first_call['items']
    current_page, total_pages = itemgetter(
        'page', 'pages')(first_call['paging'])
    print('total_pages: ', total_pages)
    while first_call['ok'] and (current_page) <= total_pages:
        try:
            print(current_page, first_call['ok'], len(message_info))
            first_call = slack_client.api_call(
                "reactions.list", user=user_id, page=current_page)
            message_info = [*message_info, *first_call['items']]
            current_page += 1
        except Exception as e:
            continue
    return message_info


user_msgs = retrieve_reaction_data(user_id)

with open('input_data/dan_bot_messages.pkl', 'wb') as f:
    pickle.dump(user_msgs, f)
