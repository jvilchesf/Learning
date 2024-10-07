import torch
import torch.nn as nn
from torch.nn import functional as F

#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O ~/input.txt

# read it in to inspect it
with open('/Users/josemiguelvilchesfierro/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Hyperparameters
batch_size = 32 # number of sequences in a mini-batch
block_size = 8  # number of characters in a sequence
max_iter = 3000 # number of training iterations
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200


list_ch = sorted(list(set(text)))
vocab_size = len(list_ch)
stoi = {ch:i for i,ch in enumerate(list_ch)}
itos = {i:ch for i,ch in enumerate(list_ch)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda h: ''.join([itos[i] for i in h])

#Encode dataset
text_encode = torch.tensor(encode(text), dtype=torch.long)
#Create a dataset
n = int(len(text_encode) * 0.9)
train = text_encode[:n]
valid = text_encode[n:]

def get_batch(split):
    data = train if split == 'train' else valid
    ix = torch.randint(len(data)- block_size, (batch_size,))
    X = torch.stack([data[   i : i+block_size   ] for i in ix])
    Y = torch.stack([data[i + 1: i+block_size +1] for i in ix])
    return X, Y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguajeModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_tokken = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets= None):        
        #Embedding tokens and targets in a 3d tensor
        logits = self.embedding_tokken(idx) # B T C [4x 8 x 65]

        #Reshape the tensor to 2d for cross entropy
        if targets is None:
            loss = None
        else:
            B , T, C = logits.shape
            logits = logits.view(B*T , C) #[32 x 65]
            targets = targets.view(B*T) #[32]
            #Calculate loss
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #get the prediction
            logits, loss = self(idx) # [B x T x C] because targets is None [4 x 8 x 65]
            #get the last token
            logits = logits[:, -1, :] # Becomes [B x C] [4 x 65]
            #softmax
            probs = F.softmax(logits, dim=1) 
            #sample
            new_token = torch.multinomial(probs, 1)
            #append to the sequence
            idx = torch.cat([idx, new_token], dim=1) # B x T+1   

        return idx

model = BigramLanguajeModel(vocab_size)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iter):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iter - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))