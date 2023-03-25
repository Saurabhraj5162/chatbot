class Vocab:
    """
    This class has methods which helps to build a voccabularies dictionary which contains all the unique
    words present in our corpus which are mapped to unique token id.
    word2idx : word -> index
    idx2word : index -> word
    WordFreq : word -> frequency of wor
    """
    sos, eos = 1,2

    def __init__(self) -> None:
        self.word2idx = {'pad':0,'sos':1, 'eos':2}
        self.idx2word = {0:'pad',1:'sos', 2:'eos'}
        self.wordFreq = {'pad':1,'sos':1, 'eos':1}
        self.counter = 3

    def insertWord(self,word):
#         try:
#             self.wordFreq[word] += 1
#         except:
#             self.word2idx[word] = self.counter
#             self.wordFreq[word] = 1
#             self.idx2word[self.counter] = word
#             self.counter +=1
        if word not in self.word2idx:
            self.word2idx[word] = self.counter
            self.wordFreq[word] = 1
            self.idx2word[self.counter] = word
            self.counter += 1
        else:
            self.wordFreq[word] += 1
            
    
    def insertListSequence(self,sequence):
        """
        This method takes a list of sequence of words and a max length integer as input argument. 
        It checks if the length of each sequence is under the defined max length or not.
        It inserts the words present in all the sequences to our vocab iff all the sequence are smaller than max_len.
        Returns: True if insertion succesful else False
        """
#        flag = True
#        print(sequence)
#        for seq in sequence:
#             if len(seq.split(' ')) >= max_len:
#                 flag = False
#        flag = len(sequence[0].split(' ')) >= max_len and len(sequence[1].split(' ')) >= max_len
        #if flag:
        for word in sequence[0].split(' '):
            self.insertWord(word)
        for word in sequence[1].split(' '):
            self.insertWord(word)
        #return flag


    def filterRareWords(self,min_freq):
        """
        This method removes all the words from vocab whosse frequence is below min_freq
        """
        words_to_keep = []
        for word, freq in self.wordFreq.items():
            if freq>=min_freq:
                words_to_keep.append(word)
        

        self.word2idx = {'pad':0,'sos':1, 'eos':2}
        self.idx2word = {0:'pad',1:'sos', 2:'eos'}
        self.wordFreq = {}
        self.counter = 3

        for word in words_to_keep:
            self.insertWord(word)

        






        


