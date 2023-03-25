import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embedding,embedding_size, hidden_size, num_layers, dropout_rate=0):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.input_sizet_size  = input_size
        #self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.embedding = embedding
        self.dropout_rate = dropout_rate
        if self.num_layers == 1: self.dropout_rate = 0
        self.gru_args = {'input_size':self.embedding_size, 'hidden_size': self.hidden_size, 
                    'num_layers': self.num_layers,'bidirectional':True, 'dropout':self.dropout_rate}
        self.gru = nn.GRU(**self.gru_args)
        
    def forward(self, query_seq, query_lengths, hidden_state = None):
        #getting the embeddings of out query
        #print(f'{1} Shape of input seq : {query_seq.size()}')
        query_seq_emb = self.embedding(query_seq)
        #print(f'{2} Shape of Emb output : {query_seq_emb.size()}')
        #we need to pack our padded sequence as it helps in reducing the computation by ignoring computation of padded token.
        query_seq_packed = nn.utils.rnn.pack_padded_sequence(query_seq_emb, query_lengths)
        
        outputs, hidden_state = self.gru(query_seq_packed, hidden_state)
        #now we need to unpack

        outputs,_ =  nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        #print('enc output size : ',outputs.size())
        
        
        #print(f'{3} Shape of Enc GRU output : {outputs.size()}')
       # print(f'{4} Shape of Enc Hidden : {hidden_state.size()}')
        
        return outputs, hidden_state
        

class Attention(nn.Module):
    def __init__(self,score, hidden_size):
        super(Attention,self).__init__()
        self.score = score
        self.hidden_size = hidden_size
        #defining two fc layers: 
        #for general and dot score
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        #for concat score (two times the input size because we concat ht and hs)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.softmax = nn.Softmax(dim=1)
        
    def getScore(self,target_hidden, source_hidden):
        """
        This method calculates the energies based on the score method passed for the luong attention mechanism.
        """
        if self.score == 'general':
            energy = self.fc1(source_hidden)
            #multiply these energies with hidden states of encoder
            weighted_hidden = energy*source_hidden
            energies = torch.sum(weighted_hidden, dim=2)
        elif self.score == 'dot':
            #energy = self.fc2(source_hidden)
            #multiply these energies with hidden states of encoder
            weighted_hidden = target_hidden*source_hidden
            energies =  torch.sum(weighted_hidden, dim=2)
            
        elif self.score == 'concat':
            #torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)
            # if target_hidden.size() != source_hidden.size():
            #     target_hidden.expand(source_hidden.size(0),-1,-1)
            
            energy = self.fc2(torch.cat((target_hidden.expand(source_hidden.size(0), -1, -1), source_hidden), 2)).tanh()

            
#             #v = v.to(device)
#             weighted_hidden = energy*v
#             energies = torch.sum(weighted_hidden, dim=2)
            energies = torch.sum(self.v * energy, dim=2)
            
        return energies.t()
        
    def forward(self,target_hidden, source_hidden):
        """
        args: target_hideen => the output of the decoder rnn
              source_hidden => the output of encoder
        """
        x = self.getScore(target_hidden, source_hidden)
        x = self.softmax(x)
        
        return x.unsqueeze(1)
        
    
    
        
    
    
class AttentionDecoder(nn.Module):
    def __init__(self, attention_method, embedding, embedding_size, hidden_size, output_size,num_layers, dropout_rate=0.2):
        super(AttentionDecoder,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers #Note: This needs to be experimented
        self.attention_method = attention_method #Experiment with it
        self.embedding_size = embedding_size
        #initialize the embedding layer
        self.embedding = embedding
        
        #initialize the GRU layer
        self.gru_args = {'input_size':self.embedding_size, 'hidden_size': self.hidden_size, 
                    'num_layers': self.num_layers}
        self.gru = nn.GRU(**self.gru_args)
        
        #initialize the fully connected layers
        self.fc1_args = {'in_features': 2*self.hidden_size, 'out_features': self.hidden_size}
        self.fc1 = nn.Linear(**self.fc1_args) 
        self.fc2_args = {'in_features': self.hidden_size, 'out_features': self.output_size}
        self.fc2 = nn.Linear(**self.fc2_args)
        
        #initialize attention:
        self.attention = Attention(self.attention_method, self.hidden_size)
        
        #final softmax
        self.softmax = nn.Softmax( dim=1)
        
    def forward(self,x, hidden_state, encoder_output):
        #shape of x : (1,batch_size)
        x_embd = self.embedding(x)
        #shape : 
        #print(f'emb shape : {x_embd.size()}')
        gru_output,curr_hidden_state = self.gru(x_embd, hidden_state)
        #print(f'dec gru op : {gru_output.size()} // hidden : {curr_hidden_state.size()}')
        #we will calculate the weights (alpha) of attention using Attention class
        alpha = self.attention(gru_output, encoder_output)
        #print('alpha size : ',alpha.size())
        #print(encoder_output.transpose(0,1).size())
        
        #now multiply these weight with encoder outputs:
        context = torch.bmm(alpha, encoder_output.transpose(0,1))
        
        #now that we have context vector and decoder output, we need to concatenate:
        x_out = torch.cat((gru_output.squeeze(0),context.squeeze(1)),1)
        #pass this x_out through a tanh activation:
        x_out = self.fc1(x_out)
        x_out = torch.tanh(x_out)
        x_out = self.fc2(x_out)
        x_out = self.softmax(x_out)
        return x_out, curr_hidden_state
                   