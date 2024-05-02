

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast
import torch.nn.functional as tnnf
import torch.optim as optim
import math
import statistics
import os

class Tokenizer:

    def traintokenizer():
        dataset = load_dataset("sentiment140")
        data1 = ''.join(str(i) for i in dataset["train"]["text"])
        data2 = ''.join(str(i) for i in dataset["test"]["text"])
        data = ''.join([data1,data2])
        special_tokens = ["<pad>","<start>","<end>"]
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train_from_iterator([data],special_tokens=special_tokens, vocab_size=30000)
        tokenizer.save("tokenizer.json")
    
    def tokenize(a):
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
        tokenized_batch = fast_tokenizer.batch_encode_plus(a)
        return tokenized_batch['input_ids']

    def tokenize_string(a):
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
        return fast_tokenizer.encode(a)

    def decode(a):
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
        return fast_tokenizer.decode(a)

def preprocess_data():
    dataset = load_dataset("sentiment140")
    trainx0 = Tokenizer.tokenize(dataset["train"]["text"])
    trainy = dataset["train"]["sentiment"]
    testx0 = Tokenizer.tokenize(dataset["test"]["text"])
    testy = dataset["test"]["sentiment"]
    trainy = [1 if x == 4 else x for x in trainy]
    testy = [1 if x == 4 else x for x in testy]
    padding_number = 0
    maximum_length = max(len(sublist) for sublist in trainx0)
    trainx = []
    testx = []
    pad_tokenized = Tokenizer.tokenize_string("<pad>")
    start_tokenized = Tokenizer.tokenize_string("<start>")
    end_tokenized = Tokenizer.tokenize_string("<end>")
    
    for i in trainx0:
        trainx.append(start_tokenized+i+pad_tokenized*(maximum_length-len(i))+end_tokenized)
    
    return maximum_length, torch.tensor(trainx), torch.tensor(trainy), torch.tensor(testx), torch.tensor(testy)

def positional_encoding(tensor_BSE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 512
    n = 10000 
    i = torch.arange(d_model/2).to(device)
    B, S, E = tensor_BSE.shape
    pos_S1 = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(1)
    
    e = E // 2
    i_e = torch.arange(e, dtype=torch.float32, device=device) 
    exponent_e = (2 * i_e) / E
    # print(e)
    denom_1e = torch.pow(1e4, exponent_e).view(1, e)
    pe_sin_Se = torch.sin(pos_S1 / denom_1e)
    pe_cos_Se = torch.cos(pos_S1 / denom_1e)
    pe_SE = torch.cat([pe_sin_Se, pe_cos_Se], dim=1).to(device)
    
    return tensor_BSE + pe_SE

def addnorm(x_BSE):
    square = torch.square(x_BSE)
    sum_BS1 = torch.mean(square, 2, keepdim=True)
    square_root_BS1 = torch.sqrt(sum_BS1)
    return x_BSE/square_root_BS1

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, ffn_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ffn = ffn_size
        self.w_1_EF = torch.nn.Parameter(torch.randn(d_model,self.ffn).to(self.device) * (d_model ** -0.5))
        self.b_1_1F = torch.nn.Parameter(torch.randn(1,self.ffn).to(self.device) * (d_model ** -0.5))
        self.w_2_FE = torch.nn.Parameter(torch.randn(self.ffn,d_model).to(self.device) * (d_model ** -0.5))
        self.b_2_1E = torch.nn.Parameter(torch.randn(1,d_model).to(self.device) * (d_model ** -0.5))

    def forward(self,tensor_BSE):
        tensor_BSE = tensor_BSE.to(self.device)
        out1 = tensor_BSE @ self.w_1_EF + self.b_1_1F
        out1 = torch.relu(out1)
        out1 = out1 @ self.w_2_FE + self.b_2_1E
        return out1


class Attention(torch.nn.Module):
    def __init__(self, d_model, Heads, Kquerys):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Kquerys = Kquerys
        self.w_EHK_k = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_v = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_q = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_o = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        
    def MultiHeadAttention(self,x_BSE):
  
        query_BSHK = torch.einsum('BSE,EHK->BSHK',x_BSE,self.w_EHK_q)
        key_BSHK = torch.einsum('BSE,EHK->BSHK',x_BSE,self.w_EHK_k)
        value_BSHK = torch.einsum('BSE,EHK->BSHK',x_BSE,self.w_EHK_v)
        logits_BSHK = torch.einsum('BSHK,BSHK->BSHK',query_BSHK, key_BSHK) / math.sqrt(self.Kquerys)
     
        softmax_BSHK = torch.softmax(logits_BSHK, dim=3)
 
        output_BSHK = torch.mul(softmax_BSHK, value_BSHK)

        a = torch.einsum('BSHK,EHK->BSE',output_BSHK,self.w_EHK_o)
        return a
class Encoder(torch.nn.Module):
    def __init__(self, Heads, ffn1):
        super().__init__()
        d_model = 512
        Kquerys = int(d_model/Heads)
        self.mha = Attention(d_model, Heads, Kquerys)
        self.ffn = FeedForward(d_model,ffn1)
    def forward(self, x_BSE):
        x_BSE = addnorm(x_BSE +  self.mha.MultiHeadAttention(x_BSE))
        x_BSE = addnorm(x_BSE +  self.ffn.forward(x_BSE))
        return x_BSE
        

class Transformer(torch.nn.Module):
    """
    B is batch size
    S is sentence length
    E is embedding size or d_model
    V is vocab size
    """
    def __init__(self):
        super().__init__()
        self.Heads = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_iterations = 300
        self.Batch_size = 5
        self.sentence_length, self.trainx, self.trainy, self.testx, self.testy = preprocess_data()
        self.d_model = 512
        self.Vocab_size = 30000
        self.embedding_matrix_VE = torch.nn.Parameter(torch.randn((self.Vocab_size, self.d_model))).to(self.device)
        self.encoder_repetitions = 5
        self.lr = 0.0001
        self.ffn = self.d_model*4
        self.encoder_layers = torch.nn.ModuleList([Encoder(self.Heads, self.ffn) for i in range(self.encoder_repetitions)])
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.linear_E1 = torch.nn.Parameter(torch.randn(self.d_model, 1).to(self.device) / math.sqrt(self.d_model))
        
    def linear(self, logits_BSE):
        logits_BV = torch.mean(logits_BSE, 1, False).to(self.device)
        out = logits_BV @ self.linear_E1
        return torch.squeeze(out)
    
    def make_batch(self, istraining):
        if istraining:
            batch = torch.randint(0, len(self.trainx) - 1, (self.Batch_size,))
            
            batch1 = self.trainx[batch]
            batch2 = self.trainy[batch]
            return batch1, batch2
        else: 
            batch = torch.randint(0, len(self.testx) - 1, (self.Batch_size,))
            batch1 = self.testx[batch]
            batch2 = self.testy[batch]
            return batch1, batch2

    def forward(self, encoder_inputs_BS):
        # take the un-embedded encoder input and put it to the device
        encoder_inputs_BS = encoder_inputs_BS.to(self.device)

        # 
        # Embed the inputs
        embedded_inputs_BSE = tnnf.embedding(encoder_inputs_BS, self.embedding_matrix_VE)

 
        # Positionally encode the inputs
        positionally_encoded_inputs_BSE = positional_encoding(embedded_inputs_BSE)

 
        #forward through the encoder layers
        for i in self.encoder_layers:
            positionally_encoded_inputs_BSE = Encoder(self.Heads, self.ffn).forward(positionally_encoded_inputs_BSE)
        # return the final output of the decoder layers
        return positionally_encoded_inputs_BSE

    def training_step(self,i):
        # make the batch of inputs for the model
        inputs_BS, targets_B = self.make_batch(True)
        inputs_BS = inputs_BS.to(self.device)
        targets_B = targets_B.to(self.device).float()
        # put the batch through the forward to produce the outputs of the model
 
        logits_BSE = self.forward(inputs_BS).to(self.device)
        logits_B = self.linear(logits_BSE).to(self.device)

        
        
        # B,S,V = logits_BSV.shape
        # put the logits into crossentropy
        # print("targets_shape",targets_B1.shape)
        # print(logits_BSV.shape)
        loss_B = tnnf.binary_cross_entropy(torch.sigmoid(logits_B),targets_B)
        # get gradients and do weight updates:
        if i%500 == 0:
            print(i,loss_B)
        self.optimizer.zero_grad() 
        loss_B.backward()# Zero gradients
          # Compute gradients
        self.optimizer.step()
        return loss_B
    def train_loop(self):
        save_interval = 1000
        saved_model_path = "model.0"
        if os.path.exists(saved_model_path):
            print("THIS EXISTS")
            self.load_state_dict(torch.load(saved_model_path))
            state_dict = torch.load(saved_model_path)
            print(f"{state_dict.keys()=}")

        for i in range(self.training_iterations):
            if i % save_interval == 0 and i > 0:
                
                torch.save(self.state_dict(), saved_model_path)
            
            self.training_step(i)
            
            
Transformer().train_loop()