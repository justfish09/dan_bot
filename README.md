## dan_bot

### The sassy, sarcastic slack bot emojifier

In honour of the most prolific emojifier we have, this project serves to pay homage to one mans efforts to add banter through slack emojis even in the darkest corners of the most boring slack channels. `dan_bot` is a slack bot that is trained to predict emoji reactions as if you had an actual Dan on your slack. Bringing you the classic favourites such as :joy: and :+1: as standard, `dan_bot` breaks the mould bringing the hits such as `squanchy`, :trollface: and `notsureif` on the regular.

This project uses the pythons `slackclient` to download all the company slack messages that have received emoji reactions from Dan. We analyse Dan's emoji usage by channel and sender over time and train a smallish, very overfitted, neural network on a sample of roughly 5500 comments vectorized using TF-IDF also taking into account the slack channel the message appeared in as well as the identity of the sender.

Full analysis and code can be found in the [notebook](https://github.com/justfish09/dan_bot/notebooks/blob/master/Basic%20Classifier.ipynb).




