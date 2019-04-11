import string
import re
import numpy as np
import torch
from torch.autograd import Variable
# from gensim.models import Word2Vec

###############################################################################
# Load data
###############################################################################

# cuda = torch.device('cuda') 
#load file and read text
def load_doc(file):
    with open(file,'r')as f:
        return f.read()
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
    token=clean_doc(re.sub('=.*=','',token))
    token=[i for i in token if i!='unk']
    return token
tokens=[]
file_name="wiki_train_tokens.txt"
tokens+=find_voc(file_name)
train_tokens=find_voc(file_name)
file_name="wiki_valid_tokens.txt"
tokens+=find_voc(file_name)
valid_tokens=find_voc(file_name)
file_name="wiki_test_tokens.txt"
tokens+=find_voc(file_name)
test_tokens=find_voc(file_name)
data_size=len(tokens)
voc=set(tokens)
voc_size=len(voc)
print(data_size)
print(voc_size)
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
    def __init__(self,num_classes,hidden_size,num_layers,dim_size,batch_size):
        super(Rnn,self).__init__()
        self.no_classes=num_classes
#         self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.batch_size=batch_size
#         self.seq_len=seq_len
        self.dim_size=dim_size
#         self.dropout_prob=drop_prob
        self.embed=nn.Embedding(voc_size,dim_size)
#         self.dropout = nn.Dropout(self.dropout_prob)
        self.lstm=nn.LSTM(dim_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(self.hidden_size,self.no_classes)

#Forward propagation of our model    
    def forward(self,x):
        embeds=self.embed(x)
        out,_=self.lstm(embeds)
#         out=out.view(-1,self.hidden_size)
        out=self.fc(out)
        return out


#Early Stopping Implementation
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.prev=np.Inf

    def call(self, val_loss, model):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} score={-score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
#         self.prev=score
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({-self.prev:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'LSTM_new.pt')
        self.val_loss_min = val_loss

###############################################################################
# Build the model
###############################################################################
vocabulary_size=voc_size
em_size=200
nhid=200
nlayers=2
epochs=100
# clip=0.25
seq_len=35
# log_interval=200
batch_size=20
decrease_indication=True
patience=6

ES=EarlyStopping(patience,decrease_indication)

model =Rnn(vocabulary_size,nhid,nlayers,em_size,batch_size)
criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
inputs,targets=get_batch(train_data,0,seq_len)
print(f'input_shape:{inputs.shape}')
output=model(inputs)
print(f'output_shape:{output.shape}')
print(f'target_shape:{targets.shape}')
loss = criterion(output.view(-1, voc_size), targets)
print(f'loss before training:{loss.item()}')


###############################################################################
# Train and Evaluate
##############################################################################
import math
def train(seq_len):
    total_loss = 0.
    start_time = time.time()
    voc_size = vocabulary_size
#     hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
        data, targets = get_batch(train_data, i,seq_len)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, voc_size), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print("loss=",loss.item())
        
def evaluate(data_source,seq_len):
    total_loss = 0.
    ntokens = vocabulary_size
#     hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data, targets = get_batch(data_source, i,seq_len)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            loss=criterion(output_flat, targets).item()
            total_loss += len(data) * loss
            print("eval_loss=",loss)
#             hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

    ###############################################################################
# Training the model
##############################################################################
import time
best_val_loss=None
try:
    for epoch in range(1, epochs+1):
        print("Epoch %d"%epoch)
        epoch_start_time = time.time()
        train(seq_len)
        val_loss = evaluate(val_data,seq_len)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        if(ES.early_stop==False):
           ES.call(val_loss,model)
        else:
           break
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open("LSTM_new.pt", 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


###############################################################################
# Test the model
##############################################################################
# with open('LSTM_new.pt', 'rb') as f:
#     model = torch.load(f)
#     # after load the rnn params are not a continuous chunk of memory
#     # this makes them a continuous chunk, and will speed up forward pass
model.lstm.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data,seq_len)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
test_loss, math.exp(test_loss)))
print('=' * 89)

###############################################################################
# Genarate the words
##############################################################################
# corpus = data.Corpus(args.data)
ntokens = vocabulary_size
# hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long)
print(i_w[input.item()])
print(input.shape)
pred_seq=""
with open("Generated.txt", 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(5):
            output = model(input)
            word_weights = output.squeeze().exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
#             print(word_idx.item())
            word = i_w[word_idx.item()]
            pred_seq+=word+" "
            outf.write(word + ('\n' if i % 20 == 19 else ' '))
            if i % 2== 0:
                print('| Generated {}/{} words'.format(i+2, 10))
print(pred_seq)