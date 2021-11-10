# generation
import torch, torch.nn as nn, torch.nn.functional as F, random, numpy as np, pdb

EON      = '<eon>'
alphabet = 'abcdefghijklmnopqrstuvwxyz'
alphabet = [char for char in alphabet] # convert to list of chars
alphabet.append(EON)

alphabet     = {v:k for k, v in enumerate(alphabet)} # convert to dict
num_to_alpha = {k:v for k, v in enumerate(alphabet)} # convert to dict
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(112)

def one_hot_encoding(letter):
    val = alphabet[letter]
    encoding = [0 for i in range(27)]
    encoding[val] = 1
    return encoding


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, dropout_prob):
        super(CharLSTM, self).__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.num_layers   = num_layers
        self.batch_size   = batch_size
        self.dropout_prob = dropout_prob
        self.lstm         = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_prob)
        self.dropout      = nn.Dropout(self.dropout_prob)
        self.linear       = nn.Linear(self.hidden_size, self.output_size)
        self.h_0          = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        self.c_0          = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x.float(), (self.h_0, self.c_0))
        x = x.reshape(-1, self.hidden_size)
        x = self.linear(x)
        return x, (h_n, c_n)


def generate(initial_char, num_names, model):
    

    names = []
    while(len(names) < num_names):

        # include initial_char
        input      = torch.tensor(one_hot_encoding(initial_char.lower()))
        input      = input.unsqueeze(0).unsqueeze(0).to(device) # unsqueeze twice to make 3-D tensor
        input_char = initial_char

        output_char    = '' # initialize to anything other than EON
        output_string  = initial_char
        MAX_OUTPUT_LEN = 15 # max_length of generated names

        while (output_char!=EON and MAX_OUTPUT_LEN > 0):
            new_input = torch.tensor(one_hot_encoding(input_char.lower()))
            new_input = new_input.unsqueeze(0).unsqueeze(0).to(device)
            input = torch.cat((input, new_input), dim=1) # append new char to the current string
            
            output, (h_n, c_n) = model(input)

            TOPK = 5
            _, idxs     = output[-1].topk(TOPK)
            idx         = random.randint(0, TOPK-1)
            pred        = idxs[idx].item()
            output_char = num_to_alpha[pred]
            if output_char != EON: # dont append eon to name
                output_string  += output_char
            
            input_char     = output_char
            MAX_OUTPUT_LEN -= 1
        
        # print(output_string)
        if output_string not in names and len(output_string) > 2: # names at least 3 chars long
            names.append(output_string)
    
    return names

# hyperparamters
hidden_size  = 512
num_layers   = 2
dropout_prob = 0.5
batch_size   = 1
num_names    = 20

saved_model_path = '0702-656377418-Garg.pt'
checkpoint       = torch.load(saved_model_path, map_location=device)
model            = CharLSTM(len(alphabet), hidden_size, len(alphabet), num_layers, batch_size, dropout_prob).to(device)
model.load_state_dict(checkpoint)
model.eval()
print(generate('a', num_names, model))
print(generate('x', num_names, model))

