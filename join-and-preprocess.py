'''
A script for aggregating abstract and metadata exports (in text format) output by the Web of Science scraper. 
Keywords are also grabbed in addition to abstracts and categories (might be useful for tagging Search Analytics abstracts) 
'''

import json
import os
from collections import Counter
import pickle
import re

# basedir = os.path.join( os.path.dirname( __file__ ), '..' )
basedir = os.path.abspath(os.path.dirname(__file__))

num_folders = 10

labels = []
abstracts = []
keywords = []

# Some Web of Science categories are assigned multiple categories (making it harder on classification model)
# May want to train only on abstracts that are assigned to a single category
multiple = 0
single = 0

def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


for x in range(1, num_folders):
		
	data_files = os.listdir(os.path.join(basedir, str(x)))
	for i, fname in enumerate(data_files):
		
		with open(os.path.join(basedir, str(x), fname), "r") as f:
			data = f.read().split("\n\n")
			for d in data:

				found_wc = False
				found_ab = False

				d = d.split("\n")
				for field in d:
					if "AB" in field[0:3]:
						found_ab = True
						abstracts.append(clean_str(field[3:]))

					if "DE" in field[0:3]:
						kw_list = field[3:].split(";")[:-1]
						for kw in kw_list:
							kw = kw.strip()
							if len(kw) > 1 and not kw[1].isupper():
								if not kw.lower() in keywords:
							 		keywords.append(kw.lower())
							elif not kw in keywords:
								keywords.append(kw)

				if found_ab == True:
					for field in d:
						if "WC" in field[0:3]:
							found_wc = True
							if ";" in field:
								multiple += 1
								labels.append(field[3:].split(";")[0]) # only capturing primary field 
							else:
								single += 1
								labels.append(field[3:])


label_counts = Counter(labels)
print json.dumps(label_counts, indent=4, sort_keys=True)

print "keywords sample: " + str(keywords[0:20])

print len(labels) == len(abstracts)
print len(labels)					

print "multiple: " + str(multiple)
print "single: " + str(single)

with open("keywords.txt", "w") as k:
	k.writelines(map(lambda x: x + '\n', keywords))

with open("abstracts.txt", "w") as a:
	a.writelines(map(lambda x: x + '\n', abstracts))

with open("labels.txt", "w") as a:
	a.writelines(map(lambda x: x + '\n', labels))

with open("allwords.txt", "w") as w:
	token_blob = ""
	for a in abstracts:
		token_blob += " " + a
	token_blob = clean_str(token_blob)
	w.write(token_blob)

	# word_indices = dict((w,i) for i,w in enumerate(token_blob.split()))
	# with open("word_indices.pkl", "wb") as out:
	# 	pickle.dump(word_indices, out, pickle.HIGHEST_PROTOCOL)




