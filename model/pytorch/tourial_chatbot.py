import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math



USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data",corpus_name)
def printLines(file, n=10):
    with open(file,'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:10]:
        print(line)
printLines(os.path.join(corpus,"movie_lines.txt"))
printLines(os.path.join(corpus,"movie_conversations.txt"))

# loadLines splits each line of the file into a dictionary of fields (lineID, characterID, movieID, character, text)
# loadConversations groups fields of lines from loadLines into conversations based on movie_conversations.txt
# extractSentencePairs extracts pairs of sentences from conversations

# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from 'loadLines' into conversations based on *movie_conversations.txy*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            # print(convObj["utteranceIDs"])
            linesIds = eval(convObj["utteranceIDs"])
            # print(linesIds)
            # Reassemble lines
            convObj["lines"] = []
            for lineId in linesIds:
                convObj["lines"].append(lines[lineId])
                # {'character1ID': 'u0', 'character2ID': 'u2', 'movieID': 'm0', 'utteranceIDs':
                # "['L194', 'L195', 'L196', 'L197']\n", 'lines': [{'lineID': 'L194', 'characterID': 'u0',
                # 'movieID': 'm0', 'character': 'BIANCA', 'text': 'Can we make this quick?  Roxanne Korrine and Andrew
                #  Barrett are having an incredibly horrendous public break- up on the quad.  Again.\n'}]}
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate overall the lines of the conversation
        for i in range(len(conversation["lines"])-1): # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# Load lines and process conversations
print("\nProcessing corpus...")
lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_FIELDS)

# Write new csv file
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)


# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)


# Defalut word tokens
PAD_token = 0 # Used for padding short sentences
SOS_token = 1 # Start-of-sentence token
EOS_token = 2 # End-of-sentence token


class Voc:
    def __init__(self,name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token:"PAD", SOS_token:"SOS", EOS_token:"EOS"}
        self.num_words = 3 # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, mincount):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k,v in self.word2count.items():
            if v >= mincount:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token:"PAD", SOS_token:"SOS", EOS_token: "EOS"}
        self.num_words = 3  # count default tokens

        for word in keep_words:
            self.addWord(word)


MAX_LENGTH = 10  # Maximum sentence length to consider


# Trun a unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])",r"\1",s)
    s = re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/reponse pairs and return a voc object
def readVocs(datafile,corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # split every line into pairs and normalize
    pairs = [ [normalizeString(s) for s in l.split('\t') ] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split('')) < MAX_LENGTH


# Filter paires using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if (filterPair(pair))]


# Using the function defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc ,pairs = readVocs(datafile,corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPair(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("counting word ...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus,corpus_name,datafile,save_dir)
# print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

MIN_COUNT = 3 # minimum word count threshold for trimming
def trimRareWords(voc, pairs, MIN_COUNT):
    # trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # only keep pairs that do not contain trimmed words in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs),len(keep_pairs),len(keep_pairs)/len(pairs)))
    return keep_pairs


# trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)


def indexesFromSentece(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')+ [EOS_token]]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentece(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# returns padded target sequence tensor,padding mask, and max target length
def outputVar(l,voc):
    indexes_batch = [indexesFromSentece(voc,sentence) for sentence in l]
    max_target_len = max(len(indexes) for indexes in indexes_batch)
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    padVar = torch.ByteTensor(padList)
    return padVar, mask, max_target_len


# returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch,voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)
"""
input_variable: tensor([[ 147,    7,   25,   25,   25],
        [  94, 1913,  247,  105,    8],
        [   7,   45,  117,  174,   40],
        [ 141,  204, 1202,    4,   60],
        [  76,  205,    7,    4,  479],
        [  37,  247, 2818,    4,    4],
        [  12,  117,    4,    2,    2],
        [1906,    7,    2,    0,    0],
        [   6,    6,    0,    0,    0],
        [   2,    2,    0,    0,    0]])
lengths: tensor([10, 10,  8,  7,  7])
target_variable: tensor([[   4,    5,   25,   56,   62],
        [   4,   37,  247,  827,    4],
        [   4,   53,  117,    7,  287],
        [  53,  920,   41,   47,  196],
        [ 317,  217,    7,    4,    4],
        [  61,    6,   68,   77,    2],
        [5191,    2,    4,  102,    0],
        [   4,    0,    2,   76,    0],
        [   2,    0,    0,    6,    0],
        [   0,    0,    0,    2,    0]])
mask: tensor([[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0]], dtype=torch.uint8)
max_target_len: 10
"""























