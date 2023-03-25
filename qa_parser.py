import csv
import random
import re
import os
import json
import itertools
import ast
import codecs

data_folder_path = 'movie-corpus/utterances.jsonl'

def printCorpus(folder, n=10):
    count = 0
    with open(folder) as corpus:
        for line in corpus:
            print(line)
            count +=1
            if count == n: break
            


#printCorpus(data_folder_path)

def parseLines(filePath):
    lines = {}
    conversations = {}

    with open(filePath,'r', encoding='iso-8859-1') as utterances:
        for line in utterances:
            #print(line)
            line_json = json.loads(line)
            
            #parsing the line
            curr_line_dict = {}
            curr_line_dict['lineID'] = line_json['id']
            curr_line_dict['characterID'] = line_json['speaker']
            curr_line_dict['text'] = line_json['text']

            lines[curr_line_dict['lineID']] = curr_line_dict

            #parsing the conversation:

            curr_conversation_id = line_json['conversation_id']

            if curr_conversation_id in conversations:
                curr_conversation_dict = conversations[curr_conversation_id]
                curr_conversation_dict['lines'].insert(0, curr_line_dict)

            else:
                curr_conversation_dict = {}
                curr_conversation_dict['conversationID'] = curr_conversation_id
                curr_conversation_dict['movieID'] = line_json['meta']['movie_id']
                curr_conversation_dict['lines'] = [curr_line_dict]
            
            conversations[curr_conversation_id] = curr_conversation_dict



    return lines, conversations

def parseQA(conversations):
    #this qa_pairs will contain lists of dialogues, 0th idx = first sentence, 1st idx = second sentence
    qa_pairs = []
    #iterate on all the lines in conversation dictionary
    for id,curr_conversation in conversations.items():
        #print(conversation['lines'])
        num_sentences = len(curr_conversation['lines'])
        for i in range(num_sentences-1):
            query = curr_conversation['lines'][i]['text'].strip()
            response = curr_conversation['lines'][i+1]['text'].strip()

            #before appending, we need to check if pair is forming or not.
            if query and response: 
                qa_pairs.append([query, response])
            
    return qa_pairs

def main():
    lines,conversations = parseLines(data_folder_path)
    pairs_qa = parseQA(conversations)

    #defining delimiter:
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    with open('training.txt', "w",encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=delimiter, lineterminator='\n')
        for qa in pairs_qa:
            writer.writerow(qa)

        
#main()
printCorpus('training.txt')


