from VocabBuilder import Vocab
from utils import *
import torch
import random

def filterPair(p,MAX_LENGTH ):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def createVocabs(filePath, max_len):
    """
    This function read txt file from filepath and fetches the QA pairs.
    After that it cleans the QA pairs and build a vocab object.
    Returns: vacab and pairs of QA
    """
    qa_pairs = []
    vocab = Vocab()
    # count =5
    #opening the txt file which contains the parsed QA pairs separated by a tab:
    with open(filePath,encoding='utf-8') as f:
        lines = f.readlines()

    #iterate on each line and get the pair and clean them using above functions:
    for qa in lines: 
        s = qa.split('\t')
        cur_pair = []
        for seq in s:
            #seq = convertUnicodeToASCII(seq)
            seq = cleanString(seq)
            cur_pair.append(seq)

        flag = filterPair(cur_pair, max_len)
        #flag = vocab.insertListSequence(cur_pair,max_len)
        #flag = vocab.insertListSequence([q,a],max_len)
        
        #check if the pair is inserted in vocab succesfully then only we will consider that pair:
        if flag: 
            qa_pairs.append(cur_pair)
            vocab.insertListSequence(cur_pair)


        # count -=1
        # if count == 0: break

    #print(qa_pairs)
    #print(vocab.word2idx)
    return vocab,qa_pairs


def filterVocab(vocab,qa_pairs, min_freq):
    """
    this function filters our vocab based on a minimum freq of words. This means rare words are filtered out.
    Also, it filters out all the QA pairs from our vocabs whose any word is rare.
    """
    #print(vocab.wordFreq)
    vocab.filterRareWords(min_freq)
    wordFreq = vocab.wordFreq
    #print(wordFreq)
    qa_pairs_filtered = []
    for qa in qa_pairs:
        q,a = qa
        if not missingWord(wordFreq,q) and not missingWord(wordFreq,a):
            qa_pairs_filtered.append(qa)
    
    # print(qa_pairs)
    # print(qa_pairs_filtered)
    return qa_pairs_filtered


def zeroPad(seq, max_len):
    """
    This function pads the sentences with 0. This is essentially required for the senetences having less words.
    """
    seq = seq.split(' ')
    while len(seq) <= max_len:
        seq.append('pad')

    return ' '.join(seq)

def getLengthsOfQa(qa_pairs):
    """
    This functions returns lists of lengths of each queries in Queries_length and lengths of each response in 
    responses_length.
    """
    queries_lengths, responses_length = [], []
    for s in qa_pairs:
        q,a = s[0],s[1]
        queries_lengths.append(len(q.split(' ')))
        responses_length.append(len(a.split(' ')))
    return queries_lengths, responses_length


def qaPairsToTensors(qa_pairs, vocab, max_len):
    """
    Takes the QA pairs :  list of lists where each list is [query, reponse]
    For each [q,r] pair creates a tensor of indexes frmo vocab and return separate lists for queries and responses
    """

    queries, responses = [],[]
    word2idx = vocab.word2idx
    for s in qa_pairs:
        #print(q,a)
        q,a = s[0], s[1]
        q += ' ' + 'eos'
        a += ' ' + 'eos'
        
        q = zeroPad(q,max_len)
        a = zeroPad(a,max_len)
        
        q_tensor = []
        a_tensor = []
        for word in q.split(' '):
            #print(word)
            q_tensor.append(word2idx[word])
        for word in a.split(' '):
            a_tensor.append(word2idx[word])
            
        #adding eos at the end:
        
            



        queries.append(q_tensor)
        responses.append(a_tensor)

    return torch.LongTensor(queries), torch.LongTensor(responses)

def getResponseMask(responses_tensor):
    """
    Takes the tensor of responses indices and creates a mask tensor.
    This mask tensor is basically populated with "1" wherever we have words and "0" whereer we have padded.
    """
    mask = []
    # if torch.is_tensor(responses_tensor):
    #     responses = responses_tensor.numpy()

    for resp in responses_tensor.numpy():
        cur_row = []
        for val in resp:
            if val == 0:
                cur_row.append(0)
            else:
                cur_row.append(1)
        mask.append(cur_row)

    return torch.BoolTensor(mask)






#max_len = 10
# min_freq = 1
#vocab,qa_pairs = createVocabs('training.txt',max_len)
# filterVocab(vocab,qa_pairs, min_freq)

# q_l,a_l = getLengthsOfQa(qa_pairs)
# print(q_l,a_l)
#quers,resps = qaPairsToTensors(qa_pairs, vocab)
# print(quers, resps)
#print(convertUnicodeToASCII('saurabh'))
# print(vocab.word2idx)

# qa_pair_batch = [random.choice(qa_pairs) for _ in range(3)]

# q_batch_tensors, q_batch_lengths, r_batch_tensors, r_batch_mask, max_r_length = createBatches(vocab, qa_pair_batch)
# print(q_batch_tensors, q_batch_lengths)
# print(r_batch_tensors, r_batch_mask)
# print(max_r_length)