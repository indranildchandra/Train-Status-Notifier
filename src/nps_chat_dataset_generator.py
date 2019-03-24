import os
import json
import nltk
import random
import re

classes_under_consideration = ['ynQuestion','whQuestion','Greet','Statement','Emotion']

out_dir = './../res/data/nps_chat_dataset'

if not os.path.exists(out_dir):
	os.makedirs(out_dir)

posts = nltk.corpus.nps_chat.xml_posts()[:]

dataset = {} 
for post in posts:
	_class = post.get('class')
	if _class not in classes_under_consideration:
		continue
	text = " "
	for word in nltk.word_tokenize(post.text):
		if not re.search('user', word, re.IGNORECASE):
			text = text + " " + word.lower()
	text = text.strip()
	if dataset.get(_class) == None:
		dataset[_class] = []
	if _class not in ['ynQuestion','whQuestion'] and len(text) > 3:
		dataset[_class].append(text)
	elif _class in ['ynQuestion','whQuestion']:
		dataset[_class].append(text)


for _class, texts in dataset.items():
	texts = random.sample(texts,533) 
	file_name = '{}.txt'.format(_class)
	with open(os.path.join(out_dir,file_name), 'w') as f:
		f.write('\n'.join(texts))