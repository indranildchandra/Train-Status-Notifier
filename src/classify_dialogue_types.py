import nltk

posts = nltk.corpus.nps_chat.xml_posts()[:]

def dialogue_act_features(post):
	features = {}
	for word in nltk.word_tokenize(post):
		features['contains({})'.format(word.lower())] = True
	return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]

size = int(len(featuresets) * 0.1)

train_set, test_set = featuresets[size:], featuresets[:size]

nb_classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(nb_classifier, test_set))