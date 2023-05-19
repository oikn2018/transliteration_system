# !pip install torchtext==0.6.0
# import locale
# locale.getpreferredencoding = lambda: "UTF-8"
# ! pip install wget
# ! pip install gdown
# ! pip install --upgrade gdown

# !pip install wandb
# ! wandb login 519ef73bbeeba4f437e82d8aeb9cf27e62a84740

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
from torch.autograd import Variable
import os
import gdown
from tqdm import tqdm
import wandb
from io import open
import string, time, math
import wget
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
from torch.utils.data import Dataset
import re
from torchtext.data import Field, TabularDataset, BucketIterator
import numpy as np
import random
import argparse


seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# CUDA
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

    
# Default configuration for best model
config = {
  "load_model": False,
  "test_model": False,
  "indic_lang": 'ben',
  "wandb_project": 'CS6910_Assignment3',
  "wandb_entity": 'dl_research',
  "dropout": 0.3,
  "learning_rate": 0.0005,
  "batch_size": 256,
  "input_embedding_size": 256,
  "num_layers": 2,
  "hidden_size": 1024,
  "cell_type": 'GRU',
  "bidirectional": True,
  "epochs": 30
}


parser = argparse.ArgumentParser()

parser.add_argument("-load","--load_model", default=config['load_model'], type=config['load_model'], required=False, help='Choose whether to load parameters of best model with attention. Choices: [True, False]', choices = [True, False])

parser.add_argument("-test","--test_model", default=config['test_model'], type=config['test_model'], required=False, help='Choose whether to test model with attention using Test Data. Choices: [True, False]', choices = [True, False])

parser.add_argument("-lang","--indic_lang", default=config['indic_lang'], type=str, required=False, help='Choose Indic Language to train and test model. Choices: ["ben", "hin"]', choices = ["ben", "hin"])

parser.add_argument("-wp","--wandb_project", default=config['wandb_project'], type=str, required=False, help='Project name used to track experiments in Weights & Biases dashboard')

parser.add_argument("-we", "--wandb_entity", default=config['wandb_entity'], type=str, required=False, help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')

parser.add_argument('-lr','--learning_rate', type=type(config['learning_rate']), default = config['learning_rate'],help=f"Choose Learning rate of the optimizer.")

parser.add_argument('-do','--dropout', type=type(config['dropout']), default = config['dropout'], help=f"Choose Dropout to be added to the Encoder and Decoder")

parser.add_argument('-bs','--batch_size', type=type(config['batch_size']), default = config['batch_size'], help=f"Choose Batch Size to be used")

parser.add_argument('-iem','--input_embedding_size', type=type(config['input_embedding_size']), default = config['input_embedding_size'], help=f"Choose Input/Output Embedding Size")

parser.add_argument('-nl','--num_layers', type=type(config['num_layers']), default = config['num_layers'], help=f"Choose Number of Layers in both Encoder and Decoder")

parser.add_argument('-hs','--hidden_size', type=type(config['hidden_size']), default = config['hidden_size'], help=f"Choose Hidden Size for both Encoder and Decoder")

parser.add_argument('-cell','--cell_type', type=type(config['cell_type']), default = config['cell_type'], help=f"Choose Cell Type: RNN, LSTM, GRU", choices = ["RNN", "LSTM", "GRU"])

parser.add_argument('-bidir','--bidirectional', type=type(config['bidirectional']), default = config['bidirectional'], help=f"Choose whether to use Unidirectional or Bidirectional Cell (Boolean Value: True/False)", choices = [True, False])

parser.add_argument('-ep','--epochs', type=type(config['epochs']), default = config['epochs'], help=f"Number of epochs for which to train model")


args = parser.parse_args()
config  = vars(args)
print(config)



# Getting the Dataset
url = 'https://drive.google.com/uc?id=1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw&export=download'

if not os.path.exists("aksharantar_sampled"):
  filename = gdown.download(url = url, quiet=False, fuzzy=True)
  print(filename)
  with ZipFile(filename, 'r') as z:
    print('Extracting files...')
    z.extractall()
    print('Done!')
  os.remove(filename)

eng_alpha = 'abcdefghijklmnopqrstuvwxyz'
pad_char = '<PAD>'

eng_alpha2idx = {pad_char: 0}
for index, alpha in enumerate(eng_alpha):
  eng_alpha2idx[alpha] = index+1



# Change Indic Language here
indic_lang = config['indic_lang']

min_range = 2304
max_range = 2431

# Bengali Unicode Hex Range: 2432-2558
# Hindi Unicode Hex Range: 2304-2431
if indic_lang == 'ben':
  min_range = 2432
  max_range = 2558
elif indic_lang == 'hin':
  min_range = 2304
  max_range = 2431

indic_alpha = [chr(alpha) for alpha in range(min_range, max_range + 1)]
indic_alpha2idx = {pad_char: 0}


for index, alpha in enumerate(indic_alpha):
  indic_alpha2idx[alpha] = index+1

indic_idx2alpha = {v: k for k, v in indic_alpha2idx.items()}
eng_idx2alpha = {v: k for k, v in eng_alpha2idx.items()}

def tokenize_indic(string):
  # return string.split()
  char_list =  [*string]
  char_list = [indic_alpha2idx[char] for char in char_list]
  return char_list

def tokenize_eng(string):
  # return string.split()
  char_list =  [*string]
  char_list = [eng_alpha2idx[char] for char in char_list]
  return char_list

import pandas as pd
  
file_names = ['test', 'train', 'valid']

for index, file_name in enumerate(file_names):
  # read contents of csv file
  file = pd.read_csv(f'aksharantar_sampled/{indic_lang}/{indic_lang}_{file_name}.csv')
    
  # adding header
  headerList = ['eng', f'{indic_lang}']
    
  # converting data frame to csv
  file.to_csv(f'aksharantar_sampled/{indic_lang}/{indic_lang}_{file_name}.csv', header=headerList, index=False)

eng = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, init_token='<sos>', eos_token='<eos>')
indic = Field(sequential=True, use_vocab=True, tokenize=tokenize_indic, init_token='<sos>', eos_token='<eos>')

fields={'eng': ('eng', eng), f'{indic_lang}': ('indic', indic)}

path_name = f'aksharantar_sampled/{indic_lang}'
train_name = f'{indic_lang}_train.csv'
val_name = f'{indic_lang}_valid.csv'
test_name = f'{indic_lang}_test.csv'
train_data, val_data, test_data = TabularDataset.splits(
    path= path_name,
    train=train_name,
    validation=val_name,
    test=test_name,
    format='csv',
    fields=fields
)


eng.build_vocab(train_data, max_size = 1000, min_freq = 1)
indic.build_vocab(train_data, max_size = 1000, min_freq = 1)

class Encoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, config=None):
    super(Encoder, self).__init__()

    if config is None:
      # Default configuration if not provided
      config = {
        "wandb_project": 'CS6910_Assignment3',
        "wandb_entity": 'dl_research',
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "batch_size": 256,
        "input_embedding_size": 256,
        "num_layers": 2,
        "hidden_size": 1024,
        "cell_type": 'GRU',
        "bidirectional": True,
        "epochs": 30
      }

    self.hidden_size = hidden_size
    self.bidirectional = config['bidirectional']
    self.num_layers = num_layers
    self.dropout = nn.Dropout(p)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.cell_type = config['cell_type']

    # Choose the appropriate recurrent cell type and configure it based on the provided parameters
    if self.cell_type == 'LSTM' and self.bidirectional:
      self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p)
    elif self.cell_type=='LSTM':
      self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    elif self.cell_type == 'GRU' and self.bidirectional:
      self.gru = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p)
    elif self.cell_type == 'GRU':
      self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p)
    elif self.cell_type == 'RNN' and self.bidirectional:
      self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p)
    elif self.cell_type == 'RNN':
      self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p)


  def forward(self, x):

    """
        Perform forward pass of the encoder.

        Args:
            x: Input sequence tensor of shape (seq_length, N) where N is the batch size.

        Returns:
            encoder_states: Encoded states for each time step of shape (seq_length, N, hidden_size).
            hidden: Final hidden state of the encoder of shape (num_layers, N, hidden_size) if unidirectional,
                    or (2*num_layers, N, hidden_size) if bidirectional.
    """

    # embedding shape: (seq_length, N, embedding_size)
    embedding = self.dropout(self.embedding(x))
    

    if self.cell_type == 'LSTM' and self.bidirectional:
      encoder_states, (hidden, cell) = self.lstm(embedding)
      hidden = (hidden[0:self.num_layers] + hidden[self.num_layers:2*self.num_layers])/2
      cell = (cell[0:self.num_layers] + cell[self.num_layers:2*self.num_layers])/2
      return encoder_states, hidden, cell
    
    elif self.cell_type == 'LSTM':
      encoder_states, (hidden, cell) = self.lstm(embedding)
      return encoder_states, hidden, cell
    
    elif self.cell_type == 'RNN' and self.bidirectional:
      encoder_states, hidden = self.rnn(embedding)
      hidden = (hidden[0:self.num_layers] + hidden[self.num_layers:2*self.num_layers])/2
      return encoder_states, hidden
    
    elif self.cell_type == 'RNN':
      encoder_states, hidden = self.rnn(embedding)
      return encoder_states, hidden
    
    elif self.cell_type == 'GRU' and self.bidirectional:
      encoder_states, hidden = self.gru(embedding)
      hidden = (hidden[0:self.num_layers] + hidden[self.num_layers:2*self.num_layers])/2
      return encoder_states, hidden
    
    elif self.cell_type == 'GRU':
      encoder_states, hidden = self.gru(embedding)
      return encoder_states, hidden

    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p, config=None):
      super(Decoder,self).__init__()

      if config is None:
        # Default configuration if not provided
        config = {
          "wandb_project": 'CS6910_Assignment3',
          "wandb_entity": 'dl_research',
          "dropout": 0.3,
          "learning_rate": 0.0005,
          "batch_size": 256,
          "input_embedding_size": 256,
          "num_layers": 2,
          "hidden_size": 1024,
          "cell_type": 'GRU',
          "bidirectional": True,
          "epochs": 30
        }
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.cell_type = config['cell_type']
      self.bidirectional = config['bidirectional']
      self.dropout = nn.Dropout(p)
      self.embedding = nn.Embedding(input_size, embedding_size)
      self.dir = 2 if self.bidirectional else 1

      if self.cell_type == 'LSTM':
        self.lstm = nn.LSTM(hidden_size*self.dir + embedding_size, hidden_size, num_layers, dropout=p)
      elif self.cell_type == 'GRU':
        self.gru = nn.GRU(hidden_size*self.dir + embedding_size, hidden_size, num_layers, dropout=p)
      elif self.cell_type == 'RNN':
        self.rnn = nn.RNN(hidden_size*self.dir + embedding_size, hidden_size, num_layers, dropout=p)

      # Linear layer to calculate attention scores
      self.energy = nn.Linear(hidden_size * (self.dir+1), 1)
      self.U = nn.Linear(hidden_size*self.dir, hidden_size)
      self.W = nn.Linear(hidden_size, hidden_size)
      self.attn = nn.Linear(hidden_size, 1)
      self.softmax = nn.Softmax(dim=0)
      self.relu = nn.ReLU()

      # Linear layer for final output prediction
      self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
      """
        Perform a forward pass of the decoder.

        Args:
            x: Input sequence tensor of shape (N), where N is the batch size.
            encoder_states: Encoded states from the encoder of shape (seq_length, N, hidden_size).
            hidden: Hidden state of the decoder of shape (num_layers*num_directions, N, hidden_size).
            cell: Cell state of the decoder for LSTM, only passed when using bidirectional LSTM. Shape is same as hidden.

        Returns:
            predictions: Output predictions of the decoder of shape (N, output_size).
            hidden: Hidden state of the decoder of shape (num_layers*num_directions, N, hidden_size).
            cell: Cell state of the decoder for LSTM, only returned when bidirectional LSTM is used. Shape is same as hidden.
      """

      # Expand dimensions to match the expected input shape
      x = x.unsqueeze(0)
      # Shape of x: (1, N)

      embedding = self.dropout(self.embedding(x))
      # embedding shape: (1,N, embedding_size)


      sequence_length = encoder_states.shape[0]
      # Implementation Logic 1: taking final layer hidden tensor value and passing it onto energy.
      hidden_temp = hidden[-1].unsqueeze(0)
      # Implementation Logic 2: taking average of all layer hidden tensor values and passing it onto energy.
      # hidden_temp = torch.mean(hidden, dim=0).unsqueeze(0)
      # Shape of hidden_temp: (1, N, hidden_size)

      # Using Baudanau Additive Attention Mechanism

      # Calculate energy scores for attention
      U = self.U(encoder_states)
      # Shape of U: (seq_length, N, hidden_size)

      # Expand the hidden states to match the sequence length
      W = self.W(hidden_temp.repeat(sequence_length, 1, 1))
      # Shape of W: (seq_length, N, hidden_size)

      energy = self.attn(torch.tanh(U+W))
#     # Shape of energy: (seq_length, N, 1)

      # Compute attention values
      attention = self.softmax(energy)
      # Shape of attention: (seq_length, N, 1)

      # Calculate the context vector using attention
      context_vector = torch.bmm(attention.permute(1, 2, 0), encoder_states.permute(1, 0, 2)).permute(1,0,2)
      # Shape of context_vector: (1, N, hidden_size)

      # Concatenate the context vector and embedded input
      rnn_input = torch.cat((context_vector, embedding), dim=2)
      # Shape of rnn_input: (1, N, hidden_size + embedding_size)

      # Pass the concatenated input through the LSTM/GRU/RNN
      if self.cell_type == 'LSTM':
        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
      elif self.cell_type == 'GRU':
        outputs, hidden = self.gru(rnn_input, hidden)
      elif self.cell_type == 'RNN':
        outputs, hidden = self.rnn(rnn_input, hidden)
      
      # Final output prediction
      predictions = self.fc(outputs)
      # Shape of predictions: (1, N, output_size)

      # Remove the first dimension to match the target shape
      predictions = predictions.squeeze(0)
      # Shape of predictions: (N, output_size)

      if self.cell_type == 'LSTM':
        return predictions, hidden, cell
      else:
        return predictions, hidden


class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, config=None):
    super(Seq2Seq, self).__init__()

    if config is None:
      # Default configuration if not provided
      config = {
        "wandb_project": 'CS6910_Assignment3',
        "wandb_entity": 'dl_research',
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "batch_size": 256,
        "input_embedding_size": 256,
        "num_layers": 2,
        "hidden_size": 1024,
        "cell_type": 'GRU',
        "bidirectional": True,
        "epochs": 30
      }

    self.run_name = 'att_cell_{}_do_{}_lr_{}_bs_{}_iem_{}_nl_{}_hs_{}_bidir_{}_ep_{}'.format(config['cell_type'], config['dropout'], config['learning_rate'], config['batch_size'], config['input_embedding_size'], config['num_layers'], config['hidden_size'], config['bidirectional'], config['epochs'])
    self.encoder = encoder
    self.decoder = decoder
    self.cell_type = config['cell_type']

  def forward(self, source, target, teacher_force_ratio = 0.5):
    """
        Perform a forward pass of the sequence-to-sequence model.

        Args:
            source: Input sequence tensor of shape (target_len, N), where N is the batch size.
            target: Target sequence tensor of shape (target_len, N).
            teacher_force_ratio: The probability of using teacher forcing during training.

        Returns:
            outputs: Output predictions of the decoder of shape (target_len, N, target_vocab_size).
    """

    batch_size = source.shape[1]
    target_len = target.shape[0]
    target_vocab_size = len(indic.vocab)

    # Initialize outputs tensor
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
    
    # Encode the source sequence
    if self.cell_type == 'LSTM':
        encoder_states, hidden, cell = self.encoder(source)
    else:
        encoder_states, hidden = self.encoder(source)

    # Initialize the first target token
    x = target[0]

    # Generate output predictions word by word
    for t in range(1, target_len):
        if self.cell_type == 'LSTM':
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
        else:
            output, hidden = self.decoder(x, encoder_states, hidden, cell=None)

        outputs[t] = output  # Adding along the first dimension (target_len)
        
        # Select the next input based on teacher forcing ratio
        best_guess = output.argmax(1)

        # Implementing Teacher Forcing
        if random.random() >= teacher_force_ratio:
           x = best_guess
        else:
           x = target[t]

    return outputs

def translit_infer(model, word, eng, indic, device, max_length=50, config = None):
    tokens = tokenize_eng(word)

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens = [eng.init_token] + tokens + [eng.eos_token]

    # Convert tokens to indices
    text_to_indices = [eng.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    word_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
      if config['cell_type'] == 'LSTM':
        encoder_states, hidden, cell = model.encoder(word_tensor)
      else:
        encoder_states, hidden = model.encoder(word_tensor)

    outputs = [indic.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_char = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
          if config['cell_type'] == 'LSTM':
            output, hidden, cell = model.decoder(previous_char, encoder_states, hidden, cell)
          else:
            output, hidden = model.decoder(previous_char, encoder_states, hidden, cell=None)
          best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Check if the model predicts the end of the sentence
        if best_guess == indic.vocab.stoi["<eos>"]:
            break

    translit_res = [indic.vocab.itos[idx] for idx in outputs]

    # remove start token
    translit_res_word = ''
    translit_res = translit_res[1:]
    # return translit_res
    for i in translit_res:
      if i != "<eos>":
        translit_res_word += indic_idx2alpha[i]
      else:
        break
    return translit_res_word

model_name=""
# Saving Checkpoint Code
def save_checkpoint(state, filename=f"{indic_lang}_{model_name}_checkpoint.pth.tar"):
    print("-x- Saving checkpoint -x-")
    torch.save(state, filename)

# Loading Checkpoint Code
def load_checkpoint(checkpoint, model, optimizer):
    print("-x- Loading checkpoint -x-")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("-x- Checkpoint Loaded Successfully! -x-")

def check_accuracy(loader, model, input_shape=None, toggle_eval=True, print_accuracy=True, config = None):
    if toggle_eval:
        model.eval()
    device = next(model.parameters()).device
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        loader.create_batches()
        for batch in loader.batches:
          for example in batch:
            num_samples += 1
            eng_word = "".join([eng_idx2alpha[val] for val in example.eng])
            indic_word = "".join([indic_idx2alpha[val2] for val2 in example.indic])
            indic_pred = translit_infer(model, eng_word, eng, indic, device, max_length=50, config = config)
            
            if indic_pred == indic_word:
              num_correct += 1

    accuracy = num_correct / num_samples
    if toggle_eval:
        model.train()
    return accuracy

### Now model is ready to train


# # Default configuration if not provided
# config = {
#   "wandb_project": 'CS6910_Assignment3',
#   "wandb_entity": 'dl_research',
#   "dropout": 0.3,
#   "learning_rate": 0.0005,
#   "batch_size": 256,
#   "input_embedding_size": 256,
#   "num_layers": 2,
#   "hidden_size": 1024,
#   "cell_type": 'GRU',
#   "bidirectional": True,
#   "epochs": 30
# }

sweep_config_w_att = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'dropout': {
            'values': [config['dropout']]
        },
        'learning_rate': {
            'values': [config['learning_rate']]
        },
        'batch_size': {
            'values': [config['batch_size']]
        },
        'input_embedding_size': {
            'values': [config['input_embedding_size']]
        },
        'num_layers': {
            'values': [config['num_layers']]
        },
        'hidden_size':{
            'values': [config['hidden_size']]
        },
        'cell_type': {
            'values': [config['cell_type']]
        },
        'bidirectional': {
            'values': [config['bidirectional']]
        },
        'epochs':{
            'values': [config['epochs']]
        }
    }
}

sweep_id_w_att = wandb.sweep(sweep_config_w_att,project=config['wandb_project'], entity=config['wandb_entity'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device being used to train/test model: {torch.cuda.get_device_name(0)}')

def train():
    torch.cuda.empty_cache()
    with wandb.init() as run:
        config = wandb.config

        # Training Hyperparameters
        num_epochs = config['epochs']
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']

        # Model Hyperparameters
        load_model = config['load_model']
        input_size_encoder = len(eng.vocab)
        input_size_decoder = len(indic.vocab)
        output_size = len(indic.vocab)
        encoder_embedding_size = config['input_embedding_size']
        decoder_embedding_size = config['input_embedding_size']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        enc_dropout = config['dropout']
        dec_dropout = config['dropout']
        cell_type = config['cell_type']
        bidirectional = config['bidirectional']

        train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size = batch_size,
        # Examples of similar length will be in same batch to minimize padding and save on compute
        sort_within_batch = True,
        sort_key = lambda x: len(x.eng),
        device = device)



        encoder_net = Encoder(
            input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout, config=config
            ).to(device)
        decoder_net = Decoder(
            input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout, config=config
            ).to(device)



        model = Seq2Seq(encoder_net, decoder_net, config=config).to(device)
        model_name = model.run_name
        run.name = model_name
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        pad_idx = indic.vocab.stoi['<pad>']
        # if all examples in batch are of similar length, don't incur penalty for this padding
        criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

        if load_model:
          load_checkpoint(torch.load(f'{indic_lang}_{model_name}_checkpoint.pth.tar'), model, optimizer)

        if indic_lang == 'hin':
          word = 'bachta'
          og_translit = 'बचता'
        elif indic_lang == 'ben':
          word = 'stagecraft'
          og_translit = 'স্টেজক্রাফট'
        acc_val_prev = 0
        acc_val_current = 0
        epoch_loss = 0

        print(f'Hyperparameter settings: {model_name}')

        for epoch in range(num_epochs):
          print(f'Epoch [{epoch+1} / {num_epochs}]')

          checkpoint = {
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()
          }
          if acc_val_current > acc_val_prev:

              if os.path.exists(f'{indic_lang}_{model_name}_checkpoint.pth.tar'):
                  os.remove(f'{indic_lang}_{model_name}_checkpoint.pth.tar')
              acc_val_prev = acc_val_current
              save_checkpoint(checkpoint, f'{indic_lang}_{model_name}_checkpoint.pth.tar')



          loop = tqdm(enumerate(train_iterator), total=len(train_iterator))
          for batch_idx, batch in loop:
            inp_data = batch.eng.to(device)
            target = batch.indic.to(device)

            output = model(inp_data, target)
            # output shape: (target_len, batch_size, output_dim)

            #basically reshape output keeping last output_dim same
            output = output[1:].reshape(-1, output.shape[2]) # so that first start token is not sent to out model
            # target -> (target_len, batch_size)
            target = target[1:].reshape(-1)
            optimizer.zero_grad()
            loss = criterion(output, target)

            loss.backward()

            # to avoid exploding gradients, clip them when they are above a threshold
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss += loss.item()

          model.eval() # turns off Dropout
          translit_res = translit_infer(model, word, eng, indic, device, max_length=50, config=config)
          print(f'Translated example word:  English: {word}, Actual: {og_translit}, Predicted: {translit_res}')
          model.train()

          print('Computing Loss and Validation Accuracy...')
          acc_val_current = check_accuracy(val_iterator, model, input_shape=None, toggle_eval=True, print_accuracy=True, config=config)
          epoch_loss = epoch_loss/len(train_iterator)

          acc_test_current = 0

          if config['test_model']:
            acc_test_current = check_accuracy(test_iterator, model, input_shape=None, toggle_eval=True, print_accuracy=True, config=config)
            metrics = {
              "loss":epoch_loss,
              "val_accuracy": acc_val_current,
              "epochs":(epoch),
              "test_accuracy": acc_test_current
              }
          else:
            metrics = {
              "loss":epoch_loss,
              "val_accuracy": acc_val_current,
              "epochs":(epoch)
              }

          wandb.log(metrics)
          if config['test_model']:
            print(f'Training Loss: {epoch_loss:.2f}, Validation Accuracy: {acc_val_current * 100:.2f}%, Test Accuracy: {acc_test_current * 100:.2f}%')
          else:
            print(f'Training Loss: {epoch_loss:.2f}, Validation Accuracy: {acc_val_current * 100:.2f}%')
          print('--------------------------')
          epoch_loss = 0


wandb.agent(sweep_id_w_att, function=train,project=config['wandb_project'], entity=config['wandb_entity'], count=15)

