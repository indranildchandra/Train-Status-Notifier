import os
import random
import tqdm
import numpy as np


data_path = './classify_text_cnn/data/mIndicator'
train_percent = 0.8

labels = {}
train_labels = {}
test_labels = {}
for file in tqdm.tqdm(os.listdir(data_path)):
	file_path = os.path.join(data_path,file)
	_class = '.'.join(file.split('.')[:-1])
	with open(file_path, 'r', encoding="utf8") as f:
		labels[_class] = f.readlines()

	a = np.arange(len(labels[_class]))
	random.shuffle(a)

	train = [labels[_class][i] for i in a[:int(train_percent*len(a))]]
	test = [labels[_class][i] for i in a[int(train_percent*len(a)):]]

	train_labels[_class] = train
	test_labels[_class] = test

if not os.path.exists('classify_text_cnn/data/mIndicatorTrain'):
	os.makedirs('classify_text_cnn/data/mIndicatorTrain')
for _class, data in train_labels.items():
	with open('classify_text_cnn/data/mIndicatorTrain/{}.txt'.format(_class), 'w') as f:
		f.write('\n'.join([i for i in data]))


if not os.path.exists('classify_text_cnn/data/mIndicatorTest'):
	os.makedirs('classify_text_cnn/data/mIndicatorTest')
for _class, data in test_labels.items():
	with open('classify_text_cnn/data/mIndicatorTest/{}.txt'.format(_class), 'w') as f:
		f.write('\n'.join([i for i in data]))

