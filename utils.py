import unicodedata
import re

def convertUnicodeToASCII(string):
    output = []
    for c in unicodedata.normalize('NFD',string):
        #checking for the egenral category assigned to the character as storng:
        uc_category = unicodedata.category(c)
        if uc_category != 'Mn': output.append(c)

    return ''.join(output)

def cleanString(string):
    string = string.lower().strip()
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    string = re.sub(r"\s+", r" ", string).strip()

    return string

def missingWord(wordMap,sequence):
    flag = False
    for word in sequence.split(' '):
        if word not in wordMap:
            print(word)
            flag = True
            break
    return flag
