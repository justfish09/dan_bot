from pathlib import Path

from dan_bot.utils import load_global_model

test_sentences =  [
    'brexit makes me sad',
    'great job on getting autocoding out, you massive nerds',
    'on leave that week',
    'production is down',
    'just be better',
    'work harder',
    'rocket to production',
    'thats just wrong',
    'windows over mac',
    'you are a bell end',
    "it's not unreasonable to have a w9am meeting",
    "My understanding from talking to different folks is the issue is due to the different text length",
    '@steven.perianen IBM is loving the new verbatim auto coding!',
    'heyhey @daniel.baark -> https://zigroup.atlassian.net/browse/SP-5320',
    "The new DS review time clashes with another meeting",
    "It's not like me to skip meals",
    "There has been a complaint about people using the putney office and keeping the door propped open. Can people make sure the door isn't kept open when it shouldn't be.",
    "Ahh we call them a Microsoft Product Team"
]

mod_path = Path(__file__).parent
path_to_file = (mod_path / '../input_data')

global_model = load_global_model(str(path_to_file / 'dan_bot.zip'))

for sentence in test_sentences:
    features = {
        'comment': [sentence],
        'channel': ['phat_data_public'],
        'user': ['donovan.thomson']
    }

    print(
        'keras model: ',
        global_model.predict_emojis(features)
    )

for sentence in test_sentences:
    features = {
        'comment': [sentence],
        'channel': ['phat_data_public'],
        'user': ['steven.perianen']
    }
    print(
        'tflite model: ',
        global_model.predict_tf_lite(features)
    )

