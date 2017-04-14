'''
A script for aggregating abstract and metadata exports (in text format) output by the Web of Science scraper. 
Keywords are also grabbed in addition to abstracts and categories (might be useful for tagging Search Analytics abstracts) 
'''

import json
import os
from collections import Counter
import pickle
import re
import argparse

# basedir = os.path.join( os.path.dirname( __file__ ), '..' )
basedir = os.path.abspath(os.path.dirname(__file__))

cs_labels = ["Computer Science, Software Engineering", "Computer Science, Cybernetics", "Computer Science, Hardware & Architecture", 
"Computer Science, Information Systems", "Computer Science, Theory & Methods", "Computer Science, Artificial Intelligence", 
"Computer Science, Interdisciplinary Applications"]

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


def num_folders():
    """
    Automatically figure out how many folders need to be looped through to get abstracts
    """

    num_folders = 0
    dir_contents = os.listdir(os.path.join(basedir, "data"))
    
    for f in dir_contents:
        try: 
            folder_num = int(f)
            if folder_num > num_folders:
                num_folders = folder_num
        except:
            pass

    return num_folders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For aggregating abstract and metadata exports (in text format) output by the Web of Science scraper')
    parser.add_argument('-n', '--number', help='max number of categories allowed to be present for an abstract', default=100000)
    args = parser.parse_args()
    args.number = int(args.number)

    print("Num folders: " + str(num_folders()))

    for x in range(1, num_folders()):
            
        data_files = os.listdir(os.path.join(basedir, "data", str(x)))
        for i, fname in enumerate(data_files):
            
            with open(os.path.join(basedir, "data", str(x), fname), "r") as f:
                data = f.read().split("\n\n")
                for d in data:

                    found_wc = False # wc = Web of Science category
                    found_ab = False # ab = Abstract
                    single_cat = False 
                    keywords_temp = [] # use to hold keywords for an individual abstract, only to be added if the category turns out to be cs
 
                    d = d.split("\n")

                    for field in d:
                        if "WC" in field[0:3]:
                            if ";" in field:
                                single_cat = False 
                            else:
                                single_cat = True
                            
                    if args.number == 1 and single_cat == False:
                        break

                    for field in d:
                        if "AB" in field[0:3]:
                            if not clean_str(field[3:]) in abstracts:
                                found_ab = True
                                abstracts.append(clean_str(field[3:]))

                        if "DE" in field[0:3]:
                            kw_list = field[3:].split(";")[:-1]
                            for kw in kw_list:
                                kw = kw.strip()
                                if len(kw) > 1 and not kw[1].isupper():
                                    if not kw.lower() in keywords_temp:
                                        keywords_temp.append(kw.lower())
                                elif not kw in keywords_temp:
                                    keywords_temp.append(kw)

                    if found_ab == True:
                        for field in d:
                            if "WC" in field[0:3]:
                                found_wc = True
                                if ";" in field:
                                    multiple += 1
                                    label = field[3:].split(";")[0].replace("\n", "")
                                    labels.append(label) # only capturing primary field 
                                    if label in cs_labels:
                                        keywords.extend(keywords_temp)
                                else:
                                    single += 1
                                    label = field[3:].replace("\n", "")
                                    labels.append(label)
                                    if label in cs_labels:
                                        keywords.extend(keywords_temp)


    label_counts = Counter(labels)
    print json.dumps(label_counts, indent=4, sort_keys=True)
    print "keywords sample: " + str(keywords[0:20])
    assert (len(labels) == len(abstracts), "Different number of abstracts and labels: %s abstracts and %s labels" %(str(len(abstracts)), str(len(labels))))
    print "multiple: " + str(multiple)
    print "single: " + str(single)
    if(args.number <= 1):
        assert(multiple == 0, "Specified a max of 1 category but abstracts with multiple categories are present in output")

    prefix = ""
    if (args.number <= 1):
        prefix = "single-category-"

    with open(os.path.join("data", prefix + "keywords.txt"), "w") as k:
        k.writelines(map(lambda x: x + '\n', keywords))

    with open(os.path.join("data", prefix + "abstracts.txt"), "w") as a:
        a.writelines(map(lambda x: x + '\n', abstracts))

    with open(os.path.join("data", prefix + "labels.txt"), "w") as a:
        a.writelines(map(lambda x: x + '\n', labels))

# with open("allwords.txt", "w") as w:
#   token_blob = ""
#   for a in abstracts:
#       token_blob += " " + a
#   token_blob = clean_str(token_blob)
#   w.write(token_blob)

    # word_indices = dict((w,i) for i,w in enumerate(token_blob.split()))
    # with open("word_indices.pkl", "wb") as out:
    #   pickle.dump(word_indices, out, pickle.HIGHEST_PROTOCOL)




