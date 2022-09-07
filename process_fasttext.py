import argparse
import os
import re
import json
import random

import gensim
import numpy as np
from tqdm import tqdm
import pandas as pd

# Padding and OOV
PAD = '<pad>'
UNK = '<unk>'
parser = argparse.ArgumentParser()
parser.add_argument("-data", "--data", type=str, default='data/Industrial.json')
parser.add_argument("-emb_dim", "--emb_dim", type=int, default=300, help="Embeddings Dimension (Default: 300)")
parser.add_argument("-tr", "--train_dataset_rate", type=float, default=0.8, help="train dataset rate")
parser.add_argument("-emb_rand_init", "--emb_rand_init", type=float, metavar="<float>", default=0.01, help="Random Initialization of Embeddings for Words without Pretrained Embeddings (Default: 0.01)")
parser.add_argument("-rs", '--random_seed', dest="random_seed", type=int, metavar="<int>", default=1337, help='Random seed (Default: 1337)')
parser.add_argument("-maxDL", "--maxDL", type=int, default=500, help="Maximum ReviewDoc Length (Default: 500)")
parser.add_argument("-vobSize", "--vobSize", type=int, default=50000, help="Maximum vocabulary Length (Default: 50000)")
args = parser.parse_args()

np.random.seed(args.random_seed)
random.seed(args.random_seed)

stopwords = open("data/stopwords_en", "r").read().split('\n')
for w in ['!', ',', '.', ';', '?', '-s', '-ly', '</s>', 's']:
    stopwords.append(w)

def simple_tokenizer(txt):
    # Convert to lowercase, remove new lines
    txt = txt.lower()
    txt = txt.replace("\r\n", " ").replace("\n", " ").replace('_', '')
    # Remove punctuation
    txt = re.sub(r"[^\w\s]", " ", txt)
    # txt = spellchecker.word_segmentation(txt).corrected_string
    # Tokenize
    filtered_words = [word for word in txt.split() if len(word) > 0 and word not in stopwords]

    return filtered_words

def getFileName(path):
    ri = path.rindex('/')
    name = path[ri + 1:-5]
    return name

def split_data_reviews(interactionJsonFilePath):
    fileName = getFileName(interactionJsonFilePath)
    dirPath = os.path.join('data/fasttext', fileName)
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
    outputItemDesPath = os.path.join(dirPath, 'item_cid.csv')
    outputTrainPath = os.path.join(dirPath, 'train.csv')
    outputTestPath = os.path.join(dirPath, 'test.csv')
    outputWordEmbeddingPath = os.path.join(dirPath, 'embedding_words.npy')
    output_nuser_nitemPath = os.path.join(dirPath, 'nuser_nitem.csv')

    interactionFile = open(interactionJsonFilePath)
    uItacList = {}
    iItacList = {}
    for line in tqdm(interactionFile, "Interaction count..."):
        if line:
            jsonLine = json.loads(line)
            asin = jsonLine['asin']
            reviewerID = jsonLine['reviewerID']
            if reviewerID not in uItacList:
                uItacList[reviewerID] = []
            if asin not in uItacList[reviewerID]:
                uItacList[reviewerID].append(asin)

            if asin not in iItacList:
                iItacList[asin] = []
            if reviewerID not in iItacList[asin]:
                iItacList[asin].append(reviewerID)

    interactionFile.close()

    interactionFile = open(interactionJsonFilePath)
    jsonId2uid={}
    jsonId2iid={}
    uidSeed=0
    iidSeed=0
    uidIidDict = {}
    iid2ReviewsTxt = {}

    train_uids = []
    train_iids = []
    test_uids = []
    test_iids = []

    for line in tqdm(interactionFile, "Data loading..."):
        if line:
            # encode user and item
            jsonLine = json.loads(line)
            asin = jsonLine['asin']
            reviewerID = jsonLine['reviewerID']
            if len(iItacList[asin]) < 5 or len(uItacList[reviewerID]) < 5:
                continue
            if asin not in jsonId2iid:
                jsonId2iid[asin] = iidSeed
                iid2ReviewsTxt[iidSeed] = ''
                iidSeed += 1
            if reviewerID not in jsonId2uid:
                jsonId2uid[reviewerID] = uidSeed
                uidSeed += 1
            # no reviewText
            if 'reviewText' in jsonLine:
                iid2ReviewsTxt[jsonId2iid[asin]] += jsonLine['reviewText'] + ' '
            uid = jsonId2uid[reviewerID]
            iid = jsonId2iid[asin]
            # every item in train set
            if iid not in train_iids:
                train_uids.append(uid)
                train_iids.append(iid)
            else:
                if uid not in uidIidDict:
                    uidIidDict[uid] = []
                if iid not in uidIidDict[uid]:
                    uidIidDict[uid].append(iid)
    interactionFile.close()

    #
    wordsCount={}
    item_words = {}
    for iid, review in tqdm(iid2ReviewsTxt.items(), "Words counting..."):
        if len(review) > 0:
            words = simple_tokenizer(review)
            item_words[iid] = []
            for word in words:
                if word not in item_words[iid]:
                    item_words[iid].append(word)
                if word not in wordsCount:
                    wordsCount[word] = 1
                else:
                    count = wordsCount[word]
                    wordsCount[word] = count + 1

    # wordId
    wordVob = sorted(wordsCount.items(), key=lambda x: x[1], reverse=True)
    word2id = {}
    word2id[PAD] = 0
    word2id[UNK] = 1
    id = 2
    for word in tqdm(wordVob, "Word numbering..."):
        word2id[word[0]] = id
        id += 1
        if id == args.vobSize:
            break

    # maxLens
    iids = []
    idDocs = []
    for iid, words in tqdm(item_words.items(), "Item-word aligning..."):
        wordIdList = []
        for word in words:
            if len(wordIdList) == args.maxDL:
                break
            if word in word2id and word2id[word] not in wordIdList:
                wordIdList.append(word2id[word])
        if len(wordIdList) < args.maxDL:
            for i in range(args.maxDL-len(wordIdList)):
                wordIdList.append(0)
        iids.append(iid)
        idDocs.append(wordIdList)
    itemDes = pd.DataFrame({'iid': iids, 'cid': idDocs})
    itemDes.to_csv(outputItemDesPath, index=False)


    for uid in tqdm(uidIidDict, "Train-Test splitting..."):
        l = len(uidIidDict[uid])
        random.shuffle(uidIidDict[uid])
        train_idx = int(l * args.train_dataset_rate)
        for i in range(l):
            if i < train_idx:
                train_uids.append(uid)
                train_iids.append(uidIidDict[uid][i])
            else:
                test_uids.append(uid)
                test_iids.append(uidIidDict[uid][i])

    num_user = []
    num_user.append(uidSeed)
    num_item = []
    num_item.append(iidSeed)

    nuser_nitem = pd.DataFrame({'num_user': num_user, 'num_item': num_item})
    nuser_nitem.to_csv(output_nuser_nitemPath, index=False)

    train_csv = pd.DataFrame({'uid': train_uids, 'iid': train_iids})
    train_csv.to_csv(outputTrainPath, index=False)

    test_csv = pd.DataFrame({'uid': test_uids, 'iid': test_iids})
    test_csv.to_csv(outputTestPath, index=False)

    initialVocabulary(word2id, wordsCount, outputWordEmbeddingPath)

    path1=os.path.join(dirPath, 'userMap.txt')
    path2 = os.path.join(dirPath, 'itemMap.txt')
    path3 = os.path.join(dirPath, 'wordMap.txt')

    f1=open(path1, 'w')
    f1.write(str(jsonId2uid))

    f2=open(path2, 'w')
    f2.write(str(jsonId2iid))

    f3=open(path3, 'w')
    f3.write(str(word2id))

    f1.close()
    f2.close()
    f3.close()

def initialVocabulary(word2id, wordsCount, outputPath):
    wid_word = {wid: word for word, wid in word2id.items()}
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('data/wiki-news-300d-1M.vec')
    wordEmbedMatrix = []

    for word, wid in tqdm(word2id.items(), "Word Embeddding..."):
        word = wid_word[wid]
        # For '<pad>' and '<unk>'
        if word in [PAD, UNK]:
            wordEmbedMatrix.append(np.zeros((args.emb_dim)).tolist())
        else:
            try:
                vec = w2v_model[word]
                # vec = w2v_model[word] / np.log2(1+wordsCount[word])
                vec = vec.tolist()
                vec = [float(x) for x in vec]
                wordEmbedMatrix.append(vec)
            except:
                # Words that do not have a pretrained embedding are initialized randomly using a uniform distribution U(−0.01, 0.01)
                vec = np.random.uniform(low=-0.01, high=0.01, size=(args.emb_dim)).tolist()
                wordEmbedMatrix.append(vec)

    wordEmbedMatrix = np.stack(wordEmbedMatrix)
    wordEmbedMatrix = np.reshape(wordEmbedMatrix, (len(word2id), args.emb_dim))
    np.save(outputPath, wordEmbedMatrix)

split_data_reviews(args.data)