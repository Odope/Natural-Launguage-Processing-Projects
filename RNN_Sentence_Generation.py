
import sentencepiece as spm
import time
import torch
import numpy as np
import torch.nn.functional as F
import random

from datasets import load_dataset

from huggingface_hub import list_datasets

from datasets import load_dataset

import math
B = 150
L = 25
lr = 0.0005
training_steps = 40

V = 32000
E = 1000
H = 1000
W_HH = torch.randn(H,H) / math.sqrt(H)
W_out_HV = torch.randn(H,V) / math.sqrt(H)
W_in_EH = torch.randn(E, H) / math.sqrt(E)
B_in_H = torch.randn(H)
B_V = torch.randn(V)

emb_VE = torch.randn(V, E)
params = [W_HH,W_in_EH,W_out_HV,B_V,B_in_H,emb_VE]
sum_loss = 0
for p in params:
    p.requires_grad = True

 
def preprocess_and_writedata():
    

    dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="train")
     
    dataset = np.asarray(dataset)
    dataset = np.ndarray.tolist(dataset)
    d1 = []
    for i in dataset:
        x = i["text"]
        d1.append(x)



     
    d1 = " ".join(d1)

    # print(type(d1))
    z = 0
    d1 = d1.replace('@-@', '')

    f = open('d1.txt','w')
    f.write(d1)
    f.close()
        


sp = spm.SentencePieceProcessor()
sp.load('oneohthree.model')





def split_data():


    f = open('d1.txt','r')
    b = f.read()
    d1 = list(b)
    f.close()

    d1 = [int(nums) for nums in d1]

     
    train = d1[(int(len(d1)*0.2)):-1]
    test = d1[:(int(len(d1)*0.1))]
    val = d1[(int(len(d1)*0.1)):(int(len(d1)*0.2))]
    return train, test, val





train, test, val = split_data()

def train():
    for bc in range(training_steps):
        hidden_state_BH = torch.zeros(B, H)
        sum_loss = 0
        start_pos_B = torch.randint(0, len(train) - L, (B,))
        for x in range(L-1):

            input_token_id_B = torch.Tensor([train[i + x] for i in start_pos_B.tolist()]).long()
            target_token_id_B = torch.Tensor([train[i + x + 1] for i in start_pos_B.tolist()]).long()

            embedded_input_BE = emb_VE[input_token_id_B]

            ev_BH = (hidden_state_BH@W_HH + embedded_input_BE@W_in_EH)

            hidden_state_BH = torch.sigmoid(ev_BH + B_in_H)

            
            logits = (hidden_state_BH @ W_out_HV) + B_V
            loss = F.cross_entropy(logits, target_token_id_B.long())
            sum_loss += loss
        for p in params:
            p.grad = None
        sum_loss.backward()

        for p in params:
            p.data += -lr * p.grad
        print("#",bc,sum_loss)







 
def run_over_val_set():
    hidden_state_BH = torch.zeros(B, H)
    sum_loss = 0
    start_pos_B = torch.randint(0, len(val) - L, (B,))
    for x in range(L-1):

        input_token_id_B = torch.Tensor([val[i + x] for i in start_pos_B.tolist()]).long()
        target_token_id_B = torch.Tensor([val[i + x + 1] for i in start_pos_B.tolist()]).long()

        embedded_input_BE = emb_VE[input_token_id_B]

        ev_BH = (hidden_state_BH@W_HH + embedded_input_BE@W_in_EH)
    
        hidden_state_BH = torch.sigmoid(ev_BH + B_in_H)
        
        
        logits = (hidden_state_BH @ W_out_HV) + B_V
        loss = F.cross_entropy(logits, target_token_id_B.long())
        sum_loss += loss
    print(sum_loss)



def sample():
    B = 1
    L = 1
    hidden_state_BH = torch.zeros(B, H)
    sum_loss = 0
    next = ""
    tokens = sp.encode(next)
    print(f"{tokens=}")
    full_string = next
    for i in range(20):
        tokenid_B = torch.tensor(tokens[i]).view(1)

        embedded_input_BE = emb_VE[tokenid_B]
    

        ev_BH = (hidden_state_BH@W_HH + embedded_input_BE@W_in_EH)

        hidden_state_BH = torch.sigmoid(ev_BH + B_in_H)

        if i + 1 < len(tokens):
            continue    


        logits = (hidden_state_BH @ W_out_HV) + B_V

        probs = F.softmax(logits, dim=1)

        
        next_token = torch.multinomial(probs, 1).item()
        tokens.append(next_token)

        print(f"{i=} {tokens=} {sp.Decode(tokens)=}")


