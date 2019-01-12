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


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropou=0):
        super(EncoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size =hidden_size
        self.embedding = embedding
        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size,n_layers,
                          dropout=(0 if n_layers==1 else dropou), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # sum bidirectional Gru outputs
        # outputs: output features from the last hidden layer of the GRU (sum of bidirectional outputs);
        # shape=(max_length, batch_size, hidden_size)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs,hidden

class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_socre(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output),
                                     2)).tanh()

    def forowrd(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_socre(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN,self).__init__()

        # keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size*2,hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model,hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output,context),1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = - torch.log(torch.gather(inp, 1, target.view(-1,1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss =loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder,decoder,embedding,
          encoder_optimizer,decoder_optimizer,batch_size,clip,max_length=MAX_LENGTH):
    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_tration else False

    # forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1,-1)
            # calculate and accumulate loss
            mask_loss, n_Total = maskNLLLoss(decoder_output,target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_Total)
            n_totals += n_Total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # calculate and accumulate loss
            mask_loss, n_Total = maskNLLLoss(decoder_output,target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_Total)
            n_totals += n_Total

    loss.backward()

    # clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm(encoder.parameters(),clip)
    _ = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()


    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optmizer, embedding,
               encoder_n_layers, decoder_n_layuers, save_dir, n_iteration, batch_size, pirnt_every, save_every,
               clip, corpus_name, loadFilename):
    # load batches for each iteration
    training_batches = [batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # initializations
    print("Initializing ...")
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # training loop
    print("Training...")
    for iteration in range(start_iteration,n_iteration+1):
        training_batch = training_batches[iteration -1]
        # extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len,
                     encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
                     batch_size, clip)
        print_loss += loss

        # print progress
        if iteration % print_every == 0:
            directory = os.path.join(save_dir,model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers,
                                                                                         decoder_n_layuers,hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iterarion':iteration,
                'en':encoder.state_dict(),
                'de':decoder.state_dict(),
                'en_opt':encoder_optimizer.state_dict(),
                'de_opt':decoder_optmizer.state_dict(),
                'loss':loss,
                'voc_dict': voc.__dict__,
                'embedding':embedding.state_dict()


            },os.path.join(directory,'{}_{}.tar'.format(iteration,'checkpoint')))


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoer_hidden = encoder_hidden[:decoder.n_layers]
        decoder_input = torch.ones(1,1,device=device, dtype = torch.long)  * SOS_token
        all_tokens = torch.zeros([0], device= device, dtype = torch.long)
        all_scores = torch.zeros([0], device = device)
        # iteratively decode one word token at a time
        for _ in range(max_length):
            # forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            # obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # record token and score
            all_tokens = torch.cat((all_tokens,decoder_input),dim=0)
            all_socres = torch.cat((all_scores, decoder_scores),dim=0)
            decoder_input = torch.unsqueeze(decoder_input,0)
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc,sentence)]
    lengths = torch.tensor[len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transopse(0,1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[tokens.item()] for token in tokens]
    return  decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('>')
            if input_sentence == 'q' or input_sentence == 'quit':break
            input_sentence = normalizeString(input_sentence)
            output_words = evaluate(encoder,decoder,searcher,voc,input_sentence)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


































