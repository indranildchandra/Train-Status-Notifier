import nltk
import os
import json
import random

classes_to_keep = ['Emotion','ynQuestion','whQuestion','Greet','Statement']

out_dir = 'classify_text_cnn/data/nps'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

posts = nltk.corpus.nps_chat.xml_posts()[:]

j = {}
for post in posts:
	_class = post.get('class')
	if _class not in classes_to_keep:
		continue
	text = post.text
	if j.get(_class) == None:
		j[_class] = []
	j[_class].append(text)

for _class, texts in j.items():
	texts = random.sample(texts,500)
	file_name = '{}.txt'.format(_class)
	with open(os.path.join(out_dir,file_name), 'w') as f:
		f.write('\n'.join(texts))