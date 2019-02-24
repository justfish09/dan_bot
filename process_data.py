import pickle
from utils import user_id
from operator import itemgetter
from collections import defaultdict


def raw_data():
	with open('input_data/dan_bot_messages.pkl', 'rb') as f:
		message_list = pickle.load(f)

	store = defaultdict(list)
	emoji_count = defaultdict(int)

	for msg in message_list:
		try:
			if 'message' in msg:
				msg_type, msg_info = itemgetter('type', 'message')(msg)
				msg_info_type, msg_text, msg_reactions = itemgetter('type', 'text', 'reactions')(msg['message'])
				if 'message' in msg:
					store[msg_text] = [reaction['name'] for reaction in msg_reactions if user_id in reaction['users']]
					for emoji in store[msg_text]:
						emoji_count[emoji] += 1
		except Exception as e:
			print(e)

	# emoji_count = dict(sorted(emoji_count.items(), key=itemgetter(1), reverse=True))
	return [store, emoji_count]
