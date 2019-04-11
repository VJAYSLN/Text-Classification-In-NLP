###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data
from random import random

from flask import Flask, render_template,request
app = Flask(__name__)

@app.route('/Form1',methods=['get'])
def Form1():
   return render_template('form1_index.html')

@app.route('/Form2')
def Form2():    
    select=request.args["selected_value"]
    # return select
    return render_template("form2_index.html",value=select)


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./LSTM_new2.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='100',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=2222,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=10,
                    help='reporting interval')
args = parser.parse_args()



import string                                                                                                                                                      
import re
import numpy as np
import torch
from torch.autograd import Variable
# from gensim.models import Word2Vec

 ###############################################################################
 # Load data
 ###############################################################################
 
#load file and read text
def load_doc(file):
  words=[]  
  with open(file,'r')as f:
    for line in f:
      line=re.sub('=.*=','',line)
      words+=line.split()
    words=[word for word in words if word!='<unk>']
  return words
#(convert into words, remove puntuations,create punctuation dict and translate,allow only alphabeticals,transform into lower case)
#(accept text corpus and return formatted words)
def clean_doc(corpus):
    tokens=corpus.split()
    table=str.maketrans('','',string.punctuation)
    tokens=[word.translate(table)for word in tokens]
    tokens=[word.lower() for word in tokens if word.isalpha()] 
    return tokens
def find_voc(file_name):
    token=load_doc(file_name)
   #token=clean_doc(re.sub('=.*=','',token))
   #token=[i for i in token if i!='unk']
    return token
tokens=[]
file_name="./data/wikitext-2/train.txt"
tokens+=find_voc(file_name)
train_tokens=find_voc(file_name)
file_name="./data/wikitext-2/valid.txt"
tokens+=find_voc(file_name)
valid_tokens=find_voc(file_name)
file_name="./data/wikitext-2/test.txt"
tokens+=find_voc(file_name)
test_tokens=find_voc(file_name)
data_size=len(tokens)
voc=set(tokens)
voc_size=len(voc)
print("Data_size",data_size)
print("Voc_size",voc_size)
#(converting each word into one_hot representation using voc_size)
# def one_hot(i):
#     one_hot_value=np.zeros((voc_size))
#     one_hot_value[i]=1
#     return one_hot_value
# print(" ".join(tokens))
#(create dictionary for word_to_index ,index_to_word,index_to_one_hot_rep)
w_i = { ch:i for i,ch in enumerate(voc)}
i_w = { i:ch for i,ch in enumerate(voc)}

# word2vec_rep={ch:w2v[i_w[ch]] for i,ch in enumerate(voc)}

###############################################################################
# Convert into Numeric_Data
###############################################################################
def numeric_tokens(tokens):
    numeric_tokens=torch.LongTensor([w_i[i] for i in tokens])
    return numeric_tokens
numeric_train_tokens=numeric_tokens(train_tokens)
numeric_valid_tokens=numeric_tokens(valid_tokens)
numeric_test_tokens=numeric_tokens(test_tokens)

###############################################################################
# Batchify dataset
###############################################################################
batch_size=20
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

eval_batch_size = 10
train_data = batchify(numeric_train_tokens,batch_size)
print(train_data.shape)
val_data = batchify(numeric_valid_tokens, eval_batch_size)
print(val_data.shape)
test_data = batchify(numeric_test_tokens, eval_batch_size)
print(test_data.shape)

###############################################################################
# Create Input And Target Sequences
###############################################################################
def get_batch(source, i,seq_len):    
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1) 
    return data, target



#Build model
#model consists of a Lstm layer and a Linear fully connected layer
###############################################################################
# Creation of model
###############################################################################
import torch.nn as nn
class Rnn(nn.Module):
    def __init__(self,num_classes,hidden_size,num_layers,dim_size,batch_size,drop_prob=0.5):
        super(Rnn,self).__init__()
        self.no_classes=num_classes
#         self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.batch_size=batch_size
#         self.seq_len=seq_len
        self.dim_size=dim_size
        self.dropout_prob=drop_prob
        self.embed=nn.Embedding(voc_size,dim_size)
        self.drop= nn.Dropout(self.dropout_prob)
        self.lstm=nn.LSTM(dim_size,hidden_size,num_layers)
        self.fc=nn.Linear(self.hidden_size,self.no_classes)
    
    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
    #Forward propagation of our model    
    def forward(self,x,hidden):
        embeds=self.embed(x)
        embeds=self.drop(embeds)
        out,hidden=self.lstm(embeds,hidden)
        out=self.drop(out)
        out=out.view(-1,self.hidden_size)
        decoded=self.fc(out)
        return decoded,hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                weight.new_zeros(self.num_layers, bsz, self.hidden_size))


if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f,map_location='cpu').cpu()


##############################n-gram model with Maximum Liklihood Estimation####################################

def generate_word(lm, history, order):
        history = history[-order:]
        his=" ".join(history)
        dist = lm[his]
        x = random()
        for w,v in dist:
            x = x - v
            if x <= 0: 
              return w
def generate_text(lm,history,order,nwords):
#     history = ["~"] * order
    history=history.split(" ")
    out = []
    for i in range(nwords):
        w = generate_word(lm, history, order)
        w=w.replace(" ","")
        w1=w.split(" ")
        history = history[-order:] + w1
        out.append(w) 
    return " ".join(out)

#########################################################################################################
import json
@app.route('/Form3')
def Form3():
    model=request.args["model"]
    text_area1=request.args["text_area1"]
    text1=request.args["text1"]
    if(model=="LSTM"):
        text_area1=text_area1.split()
        ntokens=voc_size
        # ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(1)
        # input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
        # print(corpus.dictionary.word2idx['Plot'])
        input=torch.tensor(w_i[text_area1[-1]]).view(1,1)
        pred_seq=""
        print(i_w[input.item()])
        with open(args.outf, 'w') as outf:
            with torch.no_grad():  # no tracking history
                for i in range(int(text1)):
                    output, hidden = model(input, hidden)
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)
                    word = i_w[word_idx.item()]
                    pred_seq+=word+" "	
                    outf.write(word + ('\n' if i % 20 == 19 else ' '))
                    if i % args.log_interval == 0:
                        print('| Generated {}/{} words'.format(i,int(text1)))
        print(pred_seq)
        return pred_seq
    else:
        yourdict = json.load(open("filename.csv"))
        history=text_area1
        print (generate_text(yourdict,history,35,int(text1)))
        return generate_text(yourdict,history,35,int(text1))

if __name__ == '__main__':
   app.run(debug=True)

'''
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
# print(corpus.dictionary.word2idx['Plot'])
input=torch.tensor(corpus.dictionary.word2idx['Plot']).view(1,1)

pred_seq=""
with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]
            pred_seq+=word+" "	
            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
print(pred_seq)
'''