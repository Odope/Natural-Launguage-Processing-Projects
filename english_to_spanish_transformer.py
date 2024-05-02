import torch
import pathlib
import keras
import sentencepiece as spm
import math
import torch.nn.functional as tnnf
from tokenizers import SentencePieceBPETokenizer
from tokenizers import Tokenizer
from transformers import AutoTokenizer
import transformers
from os import path
import os
import torch.optim as optim
from transformers import PreTrainedTokenizerFast
import random


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tokenizer:
    def __init__(self):
        self.saved_directory = "RTC:Untitled Folder/"


    def traintokenizer(self):
        text_file_path = "transformer_piece.txt" 
        special_tokens = ["<pad>", "<start>", "<end>"]

        tk_tokenizer = SentencePieceBPETokenizer()

        # Read text data from the file
        with open(text_file_path, 'r') as file:
            text_data = file.readlines()

        # Train tokenizer from the text data
        tk_tokenizer.train_from_iterator(
            text_data,
            vocab_size=30000,
            min_frequency=2,
            show_progress=True,
            special_tokens=special_tokens
        )


        # print("Tokenizer directory:", self.saved_directory)
        # print("Vocabulary file path:", os.path.join(self.saved_directory, "vocab.json"))
        # print("Merges file path:", os.path.join(self.saved_directory, "merges.txt"))

        # Create the directory to save tokenizer files if it doesn't exist
        if not os.path.exists(self.saved_directory):
            os.makedirs(self.saved_directory)

        # Save the tokenizer files
        tk_tokenizer.save("tokenizer.json")




    def tokenize(self,a):
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
        tokenized_batch = fast_tokenizer.batch_encode_plus(a)

# Print tokenized input IDs
        return tokenized_batch['input_ids']






    def tokenize_string(self,a):
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
        return fast_tokenizer.encode(a)

    def decode(self,a):
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
        return fast_tokenizer.decode(a)




class getDataset:

    def __init__(self):
        pass

    def data():


        text_file = keras.utils.get_file(
            fname="spa-eng.zip",
            origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
            extract=True,
        )
        text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"

        # class embedding:

        #     def __init__(self,vocab_size,)

        with open(text_file) as f:
            raw2 = f.read()
        raw = raw2.split('\n')[:-1]

        r = open('transformer_piece.txt','w')
        r.write(str(raw2))
        r.close()

        # print("done 10")
        # print(raw)
        elist = []
        slist = []
        pairs_text = []
        for line in raw:
            # print(line)
            e,s = line.split('\t')
           
            pairs_text.append(e)
            pairs_text.append(s)

            elist.append(e)
            slist.append(s)

        # print(elist[:10])
        # print(type(elist))
        elist = Tokenizer().tokenize(elist)
        # print("type", type(elist))
        slist = Tokenizer().tokenize(slist)
        e1list = []
        s1list = []
        # print("done 20")

        for i in range(len(elist)):
            # print(elist[i])
            e1list.append(len(elist[i]))
            s1list.append(len(slist[i]))

        padding_num_e = max(e1list)
        padding_num_s = max(s1list)
 

        # print("padding",padding_num_e)
        Sentence_length = padding_num_e
        # print("done 30")
        pairs_text2 = Tokenizer().tokenize(pairs_text)
        # print("done 35")
        # print(pairs_text[0])
        datax = []
        datay = []
        data_targets = []
        pad_tokenized = Tokenizer().tokenize_string("<pad>")
        # print(pad_tokenized)
        # print(type(pad_tokenized))
        end_tokenized = Tokenizer().tokenize_string("<end>")
        start_tokenized = Tokenizer().tokenize_string("<start>")
        # print("pad tokenized",pad_tokenized)
        for i in range(0,len(pairs_text2),2):
            # print(i)

            a = pairs_text2[i] + pad_tokenized * (padding_num_e-len(pairs_text2[i]))
            # print(a)
            # print(a[:10])
            # print(len(prin[i][1]))
            # print(padding_num_s)
            b = start_tokenized + pairs_text2[i+1] + pad_tokenized * (padding_num_s-len(pairs_text2[i+1]))
            c = pairs_text2[i+1] + end_tokenized + pad_tokenized * (padding_num_s-len(pairs_text2[i+1]))
            
            if i == 0:
                (len(b))
                (len(c))
            
            datax.append(a)
            datay.append(b)
            data_targets.append(c)
            # (i, len(pairs_text2))
        # ("done 40")

        # (data2[0])
        # FINISH PADDING 


        # (len(pairs_text))

        # (padding_num_e)
        trainx, trainy = datax[0:int(len(datax)*0.8)], datay[0:int(len(datay)*0.8)]
        train_targets = data_targets[0:int(len(data_targets)*0.8)]
        ("targets",len(data_targets[0]))
        valx, valy = datax[int(len(datax)*0.8): int(len(datax)*0.9)], datay[int(len(datay)*0.8): int(len(datay)*0.9)]
        val_targets = data_targets[int(len(data_targets)*0.8): int(len(datax)*0.9)]
        testx, testy = datax[int(len(datax)*0.9): -1], datay[int(len(datay)*0.9): -1]
        test_targets = data_targets[int(len(data_targets)*0.9): -1]
        
        
        # (trainx[:5])
        # b = open('transformer_piece.txt','r')
        # d = b.read()
        # (d[:10])
        # b.close()
        # (trainy[0])
        trainx = torch.tensor(trainx)
        trainy = torch.tensor(trainy)
        valx = torch.tensor(valx)
        valy = torch.tensor(valy)
        testx = torch.tensor(testx)
        testy = torch.tensor(testy)
        train_targets = torch.tensor(train_targets)
        (f"{train_targets.shape=}")
        test_targets = torch.tensor(test_targets)
        val_targets = torch.tensor(val_targets)
        print("test_targets",Tokenizer().decode(test_targets[0]))
        return trainx, trainy, train_targets, valx, valy, val_targets, testx, testy, test_targets, Sentence_length



class InputEmbedding:
    # PASS A 2D ARRAY PLZ LOLZ
    def __init__(self, Vocab_size, d_model, encoder_embedding, decoder_embedding):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_embedding = encoder_embedding.to(self.device)
        self.decoder_embedding = decoder_embedding.to(self.device)
    def encoder_embed(self,Encoded_SB):
        Encoded_SB = torch.tensor(Encoded_SB).to(self.device)
        # global emb_SBE
        # (self.embeddingA(Encoded_SB).size())
        return self.encoder_embedding(Encoded_SB)
    def decoder_embed(self, Encoded_SB):
        Encoded_SB = torch.tensor(Encoded_SB).to(self.device)
        return self.decoder_embedding(Encoded_SB)


class PositionalEncoding:
    def positonalencoding(x_BSE):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_BSE = x_BSE.to(device)
        (f"{x_BSE.shape=}")
        B, S, E = x_BSE.shape
        pos_S1 = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(1)
        assert E % 2 == 0
        e = E // 2
        i_e = torch.arange(e, dtype=torch.float32, device=device)
        exponent_e = (2 * i_e) / E
        denom_1e = torch.pow(1e4, exponent_e).view(1, e)
        pe_sin_Se = torch.sin(pos_S1 / denom_1e)
        pe_cos_Se = torch.cos(pos_S1 / denom_1e)
        pe_SE = torch.cat([pe_sin_Se, pe_cos_Se], dim=1).to(device)
        return x_BSE + pe_SE



class MultiheadAttention(torch.nn.Module):
    
    def __init__(self, d_model, Heads, Kquerys, use_diagonal_mask=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Kquerys = Kquerys
        self.w_EHK_k = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_v = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_q = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_o = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.use_diagonal_mask = use_diagonal_mask
        

    # def params(self):
    #     return self.w_EHK_k, self.w_EHK_v, self.w_EHK_q, self.w_EHK_o

    def forward(self, x_BSE, encoder_output_BSE=None):
        query_BSHK = torch.einsum('BSE,EHK->BSHK', x_BSE, self.w_EHK_q)
        m_BSE = encoder_output_BSE if encoder_output_BSE is not None else x_BSE
        value_BMHK = torch.einsum('BSE,EHK->BSHK', m_BSE, self.w_EHK_v)
        key_BMHK = torch.einsum('BSE,EHK->BSHK', m_BSE, self.w_EHK_k)

        logits_BSHM = torch.einsum('BSHK,BMHK->BSHM',query_BSHK, key_BMHK) / math.sqrt(self.Kquerys)
        if self.use_diagonal_mask:
            B, S, H, M = logits_BSHM.shape
            query_pos_1S11 = torch.arange(S, device=logits_BSHM.device).view(1, S, 1, 1)
            memory_pos_111M = torch.arange(M, device=logits_BSHM.device).view(1, 1, 1, M)
            visiBSE_1S1M = query_pos_1S11 >= memory_pos_111M
            mask_1S1M = torch.where(visiBSE_1S1M, 0, -torch.inf)
            logits_BSHM = logits_BSHM + mask_1S1M
        softmax_BSHM = torch.softmax(logits_BSHM, dim=3)
        output_BSHK = torch.einsum('BLHM,BMHK->BLHK', softmax_BSHM, value_BMHK)
        out_BSE = torch.einsum('BSHK,EHK->BSE', output_BSHK, self.w_EHK_o)
        return out_BSE


class FeedForward(torch.nn.Module):
    def __init__ (self, d_model,ffn):
        super().__init__()
        self.ffn = ffn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w_1_EF = torch.nn.Parameter(torch.randn(d_model,self.ffn).to(self.device) * (d_model ** -0.5))
        self.b_1_1F = torch.nn.Parameter(torch.randn(1,self.ffn).to(self.device) * (d_model ** -0.5))
        self.w_2_FE = torch.nn.Parameter(torch.randn(self.ffn,d_model).to(self.device) * (d_model ** -0.5))
        self.b_2_1E = torch.nn.Parameter(torch.randn(1,d_model).to(self.device) * (d_model ** -0.5))

    # def params(self):
    #     return self.w_1_EF, self.w_2_FE, self.b_1_1F, self.b_2_1E

    def forward(self, tensor_BSE):
        # logits_BSD = self.w_2 @ (torch.max(torch.tensor(0), (self.w_1 @ tensor_BSE) + self.b_1)) + self.b_2
        hidden_BSF = tensor_BSE @ self.w_1_EF + self.b_1_1F
        hidden_BSF = torch.relu(hidden_BSF)
        output_BSE = hidden_BSF @ self.w_2_FE + self.b_2_1E
        return output_BSE
 # return logits_BSD


def rms_norm(x_BSE):
    square = torch.square(x_BSE)
    sum2_BS1 = torch.mean(square, 2, keepdim=True)
    square_root_BS1 = torch.sqrt(sum2_BS1)
    return x_BSE/square_root_BS1


# class LinearSoftmax:
# def __init__(self):
# pass
# def linear(self, addnormoutput):

class EncoderLayer(torch.nn.Module):

    def __init__(self, d_model, Heads, ffn1):
        super().__init__()
        Kquerys = int(d_model/Heads)
        self.mha = MultiheadAttention(d_model, Heads, Kquerys)
        self.ffn = FeedForward(d_model,ffn1)

    # def params(self):
    #     return self.mha.params() + self.ffn.params()

    def forward(self, x_BSE):        
        x_BSE = rms_norm(x_BSE +  self.mha.forward(x_BSE))
        x_BSE = rms_norm(x_BSE +  self.ffn.forward(x_BSE))
        return x_BSE


class DecoderLayer(torch.nn.Module):

    def __init__(self, d_model, Heads, ffn1):
        super().__init__()
 
        self.Kquerys = int(d_model/Heads)
  
        # self.encoder_layers = []
        self.self_attention = MultiheadAttention(d_model, Heads, self.Kquerys, use_diagonal_mask=True)
        self.encoder_decoder_attention = MultiheadAttention(d_model, Heads, self.Kquerys)
        self.ffn = FeedForward(d_model, ffn1)

    def params(self):
        return self.self_attention.params() + self.encoder_decoder_attention.params() + self.ffn.params()

    def forward(self, x_BSE, encoder_output_BSE):
        x_BSE = rms_norm(x_BSE +  self.self_attention.forward(x_BSE))
        x_BSE = rms_norm(x_BSE +  self.encoder_decoder_attention.forward(x_BSE, encoder_output_BSE))
        x_BSE = rms_norm(x_BSE +  self.ffn.forward(x_BSE))
        return x_BSE

       
# class Linear(torch.nn.Module):
#     def __init__(self, Embedding_Matrix_VE, Vocab_size, d_model):
#         super().__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.V, self.E = Vocab_size, d_model
#         # (type(self.decoder_output_BSE))
#         self.embedding_matrix_VE = Embedding_Matrix_VE.to(self.device)
#     def params(self):
#         return self.bias
#     def linear(self, decoder_output_BSE):
#         self.decoder_output_BSE = decoder_output_BSE.to(self.device)
#         # print("size",self.decoder_output_BSE.size())
#         # print(self.embedding_matrix_VE.weight.size())
#         self.logits = (self.decoder_output_BSE @ self.embedding_matrix_VE.t()) + self.bias
#         return self.logits

class Softmax:
    def __init__(self, linear_output_BSV):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_output_BSV = linear_output_BSV.to(self.device)
    def softmax(self):
        return tnnf.softmax(self.linear_output_BSV)

class transformerGUI:
    def __init__(self):
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("Text Input")
        self.root.geometry("400x250")

        frame = tk.Frame(self.root, borderwidth=2, relief="ridge")
        frame.pack(pady=10)


        self.entry = tk.Entry(self.root, width=40)
        self.entry.pack(pady=10)

        # Create a button
        self.button = tk.Button(self.root, text="Translate", command=self.transformerGUI_input())
        self.button.pack(pady=(0, 10))  # Pack the button with bottom padding

        # Create a label inside the frame with text wrapping
        self.label = tk.Label(self.frame, text="Type something in the text box below to translate it into Spanish!", font=("Helvetica", 10), fg="black", wraplength=380)
        self.label.pack(padx=10, pady=5)  # Pack the label inside the frame

        self.root.mainloop()

    def transformerGUI_input(self):
        text = self.entry.get()
        self.label.config(text=text)
    def transformerGUI_output(self):
        pass
    

class Transformer(torch.nn.Module):
    def __init__(self, lr = 0.001, Vocab_size = 30000, d_model = 512, Batch_size = 8000, Sentence_length = 0, Heads = 8, training_iterations = 1, encoder_repetitions = 6, decoder_repetitions = 6):
        super().__init__()
        self.lr = lr
        self.Vocab_size = Vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.Batch_size = Batch_size
        self.Sentence_length = Sentence_length
        self.ffn = d_model*4
        self.Heads = Heads
        self.Kquerys = int(d_model/Heads)
        self.embedding_VE = torch.nn.Parameter(torch.randn(self.Vocab_size, self.d_model, device=DEVICE) * (d_model ** -0.5))
        self.training_iterations = training_iterations
        self.encoder_repetitions = encoder_repetitions
        self.decoder_repetitions = decoder_repetitions
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(self.d_model, self.Heads, self.ffn) for i in range(encoder_repetitions)])
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(self.d_model, self.Heads, self.ffn) for i in range(decoder_repetitions)])
        
        
        self.trainx, self.trainy, self.train_targets, self.valx, self.valy, self.val_targets, self.testx, self.testy, self.test_targets, self.Sentence_length = getDataset.data()

        # multiply by insverse sqrt of first dimention
        
        # self.linear_layer = Linear(self.embedding_VE, self.Vocab_size, self.d_model)
        
        # self.encoder_layers = []
        # self.input_embeding = InputEmbedding(self.Vocab_size, self.d_model)

        # get all trainable params!!!!

        # for i in self.params():
        #     # print(i)
        #     i.requires_grad_(True)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    # def params(self):
    #     ret = [self.linear_layer.params()]
    #     for i in self.encoder_layers + self.decoder_layers:
    #         ret.extend(i.params())
    #     return ret


    def make_batch(self, enc_inp, dec_inp, dec_tgt):
        def _len_without_padding(x):
            return (x > 0).to(torch.int32).sum()

        def _max_len(idx):
            return max([
                _len_without_padding(enc_inp[idx]),
                _len_without_padding(dec_inp[idx]),
                _len_without_padding(dec_tgt[idx])])
        
        start_pos = random.randint(0,enc_inp.size(0))
        L = _max_len(start_pos)
        B = self.Batch_size // L
        B = min(B, enc_inp.size(0) - start_pos)
        return (
            enc_inp[start_pos:start_pos+B, :L], 
            dec_inp[start_pos:start_pos+B, :L], 
            dec_tgt[start_pos:start_pos+B, :L]
        )
    
    def make_training_batch(self, train, val, test):
        if train:
            return self.make_batch(
                self.trainx, self.trainy, self.train_targets
            )
        elif val:
            return self.make_batch(
                self.valx, self.valy, self.val_targets
            )
        elif val:
           return self.make_batch(
                self.testx, self.testy, self.test_targets
            )
        
        if train:
            self.batch = torch.randint(0, len(self.trainx) - 1, (self.Batch_size,))
            encoder_inputs_BL = self.trainx[self.batch]
            decoder_inputs_BL = self.trainy[self.batch]
            decoder_targets_BL = self.train_targets[self.batch]
        if val:
            self.batch = torch.randint(0, len(self.valx) - 1, (self.Batch_size,))
            encoder_inputs_BL = self.valx[self.batch]
            decoder_inputs_BL = self.valy[self.batch]
            decoder_targets_BL = self.val_targets[self.batch]
        if test:
            self.batch = torch.randint(0, len(self.testx) - 1, (self.Batch_size,))
            encoder_inputs_BL = self.testx[self.batch]
            decoder_inputs_BL = self.testy[self.batch]
            decoder_targets_BL = self.test_targets[self.batch]
        return encoder_inputs_BL, decoder_inputs_BL, decoder_targets_BL
        

    def encoder_forward(self, encoder_inputs_BL):
        encoder_inputs_BL = encoder_inputs_BL.to(self.device)
        x_BSE = tnnf.embedding(encoder_inputs_BL, self.embedding_VE)
        positional_encoded_BSE = PositionalEncoding.positonalencoding(x_BSE)
        for encoder_layer in self.encoder_layers:
            positional_encoded_BSE = encoder_layer.forward(positional_encoded_BSE)
        return positional_encoded_BSE

    def decoder_forward(self, encoder_output_BSE, decoder_inputs_BL):
        x_BSE = tnnf.embedding(decoder_inputs_BL.to(self.device), self.embedding_VE.to(self.device)).to(self.device)
        x_BSE = PositionalEncoding.positonalencoding(x_BSE).to(self.device)
        for decoder_layer in self.decoder_layers:
            x_BSE = decoder_layer.forward(x_BSE, encoder_output_BSE)
        logits_BSV = (x_BSE @ self.embedding_VE.t()).to("cuda")
        return logits_BSV
    
    def forward(self, encoder_inputs_BL, decoder_inputs_BL):
        encoder_output_BSE = self.encoder_forward(encoder_inputs_BL)
        decoder_forward = self.decoder_forward(encoder_output_BSE, decoder_inputs_BL)
        return decoder_forward

    
        


    def train_step(self, train, val, test):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_inputs_BL, decoder_inputs_BL, decoder_targets_BL = self.make_training_batch(train, val, test)
        # print(f"{decoder_targets_BL.shape=}")
        # print(f"{decoder_inputs_BL.shape=}")
        # print(f"{encoder_inputs_BL.shape=}")
        # decoder_targets_BL = torch.squeeze(tnnf.embedding(decoder_targets_BL, self.embedding_VE))
        # print(decoder_targets_BL.shape)
        decoder_targets_BL = decoder_targets_BL.to(device)
        logits_BSV = self.forward(encoder_inputs_BL, decoder_inputs_BL).to(device)
        # print("logits", logits_BSV.shape)
        # print("decoder", decoder_targets_BL.shape)
        B, L, V = logits_BSV.shape
        loss_BL = tnnf.cross_entropy(
            logits_BSV.view(B * L, V),
            decoder_targets_BL.view(B * L).to(device),
            reduction='none',
        ).view(B, L)
        is_not_padding_BL = (decoder_targets_BL > 0).float()
        loss_BL = loss_BL * is_not_padding_BL
        loss = torch.sum(loss_BL) / torch.sum(is_not_padding_BL)
        a = loss
        # print(a)
        self.optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()
        return a
        # self.encoderembeddingmatrix.weight.grad.zero_()
        # self.decoderembeddingmatrix.weight.grad.zero_()

    def train(self):
        save_interval = 500
        x = 250
        saved_model_path = "saved_model/model.16"
        if os.path.exists(saved_model_path):
            print("THIS EXISTS")
            self.load_state_dict(torch.load(saved_model_path))
            state_dict = torch.load(saved_model_path)
            print(f"{state_dict.keys()=}")
        losses = []
        for z in range(self.training_iterations):
            if z % save_interval == 0 and z > 0:
                torch.save(self.state_dict(), saved_model_path)
            
            if z % x == 0:
                print("training_step", z)
                losses = torch.tensor(losses)
                print(torch.sum(losses)/x)
                losses = []
            a = self.train_step(True, False, False)
            losses.append(a)
            
            # self.batch = torch.randint(0, len(self.trainx) - 1, (1, self.Batch_size))
            # # print(f"{self.train_targets.shape=}")
            # self.train_targets_1 = torch.squeeze(self.train_targets[self.batch]).to("cuda")
            # Encoder_Instance = EncoderLayer(self.batch, self.trainx, self.Vocab_size, self.d_model, self.Batch_size, self.Sentence_length, self.Heads, self.training_iterations, self.encoder_repetitions, self.decoder_repetitions, self.input_embedding, self.mha, self.ffn)
            # Encoder_Output = Encoder_Instance.forward()
            # Decoder_Instance = DecoderLayer(self.batch, self.trainy, Encoder_Output, self.Vocab_size, self.d_model, self.Batch_size, self.Sentence_length, self.Heads, self.training_iterations, self.encoder_repetitions, self.decoder_repetitions, self.input_embedding, self.mha, self.ffn)
            # Decoder_Ouput = Decoder_Instance.forward()
            # for i in range(self.encoder_repetitions-1):
            #     Encoder_Output = Encoder_Instance.forward()
            # for i in range(self.decoder_repetitions-1):
            #     Decoder_Ouput = Decoder_Instance.forward()
            # # print(type(Decoder_Ouput))
            
            
            # print("training_iteration:",z)


    def test_val(self):
        self.train_step(False, True, False)
        
    # FINISH SAMPLING CODE LOL


    def test_test(self):
        self.train_step(False, False, True)
        
        
    
    def sample(self, input_sentence):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = Tokenizer().tokenize_string(input_sentence)
        encoder_input = torch.tensor(input).to(device)
        print(encoder_input.shape)
        encoder_input = encoder_input.unsqueeze(0)
        print(encoder_input.shape)
        generate = True
        encoder_forward = self.encoder_forward(encoder_input)
        decoder_input_list = Tokenizer().tokenize_string("<start>")
        while generate:
            tokenized_sentence = torch.tensor(decoder_input_list).to(device)
            tokenized_sentence = tokenized_sentence.unsqueeze(0)
            decoder_input_BL = torch.tensor(decoder_input_list).view(1, -1).to(device)
            logits_BSV = self.decoder_forward(encoder_forward, decoder_input_BL)
            # decoder_input = logits_BSV
            # print(logits_BSV.shape)
            logits_V = logits_BSV[0, -1].to(device)
            # print(logits_V)
            # print(logits_V.shape)
            argmax = torch.argmax(logits_V).item()
            decoder_input_list.append(argmax)
            print(f"{decoder_input_list=}")
            if len(decoder_input_list) > 10:
                break
            
            


        print(Tokenizer().decode(decoder_input_list))
        

        
 
# print("done")
a = Transformer()
a.train()
# a.test_val()
# a.test_test()
# a.test_val()
# a.test_test()
while True:
    abc = input()
    a.sample(abc)
# a.test_test()
# a.test_val()