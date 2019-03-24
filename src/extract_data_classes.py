import json
import os
import nltk
import tqdm
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

data_path = './../res/data/mIndicator_dataset/raw_data/train'
key_data_path = './../res/data/mIndicator_dataset/raw_data/class_definitions'
out_path = './../res/data/mIndicator_dataset/labelled_data/labelled_train_data.json'

stemmer = EnglishStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tagger = nltk.data.load("taggers/maxent_treebank_pos_tagger/english.pickle")
Wh_tags = ['WP','WRB','WDT','WP$']

labeled_data = {}
key_data = {}

def isQuestion(sent):
	sent = sent.lower()
	tokens = nltk.word_tokenize(sent)
	tags = tagger.tag(tokens)

	if '?' in sent:
		return True

	for tag in tags:
		if tag[1] in Wh_tags:
			return True
	
	if (len(tokens)>=2 and tokens[-2]=='or' and tokens[-1]=='not') or (len(tokens)>=3 and tokens[-3]=='or' and tokens[-2]=='not'):
		return True

	if (len(tags)>=3 and tags[0][1] in ['VBZ'] and tags[1][1] in ['RB','RBR','RBS'] and tags[2][1] in ['DT','EX']) or\
		(len(tags)>=2 and tags[0][1] in ['VBZ'] and tags[1][1] in ['DT','EX']):
		return True

	return False


for file in os.listdir(key_data_path):
	file_path = os.path.join(key_data_path,file)
	with open(file_path, 'r', encoding="utf8", errors='ignore') as f:
		lines = f.readlines()

	_class = '.'.join(file.split('.')[:-1])
	key_data[_class] = []
	for line in lines:
		stemmed = " ".join([stemmer.stem(w) for w in word_tokenize(line)])
		key_data[_class].append(stemmed)

for file in tqdm.tqdm(os.listdir(data_path)):

	file_path = os.path.join(data_path,file)

	with open(file_path, 'r', encoding="utf8", errors='ignore') as f:
		print("Reading file - " + file_path + " ...")
		try:
			data_json = json.load(f)
		except Exception as e:
			print("Exception while reading file - " + file_path + " !")
			print(str(e))
			data_json = {}

			# # Find the offending character index:
			# idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))
			# print("Position of invalid character - " + idx_to_replace)
			# data = open(file_path, "r").read()
			# # Remove the offending character:
			# data_json = list(data)
			# data_json[idx_to_replace] = ' '
			# new_message = ''.join(data_json)
			# data_json = new_message
			# print("Data: ------------------")
			# print(json.dumps(data_json, indent=4, sort_keys=True))
			
	messages_arr_stemmed = []
	if bool(data_json):
		for timestamp, data in data_json.items():
			msg_lemmed = " ".join([wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(data['m'])])
			msg_stemmed = " ".join([stemmer.stem(msg) for msg in word_tokenize(msg_lemmed)])
			messages_arr_stemmed.append([msg_stemmed,timestamp])

		for stemmed_msg, timestamp in messages_arr_stemmed:
			if isQuestion(data_json[timestamp]['m']):
				if labeled_data.get('questions') is None:
					labeled_data['questions'] = []
				labeled_data['questions'].append(data_json[timestamp])
			else:
				for _class, _ in key_data.items():
					for keyword in key_data[_class]:
						if keyword in stemmed_msg:
							if labeled_data.get(_class) is None:
								labeled_data[_class] = []
							labeled_data[_class].append(data_json[timestamp])


with open(out_path, 'w', encoding="utf8", errors='ignore') as f:
	print("Dumping output to JSON file...")
	json.dump(labeled_data, f)