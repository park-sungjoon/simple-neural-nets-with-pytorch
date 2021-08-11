import torch
import torch.nn as nn
import numpy as np
import numpy.random as rand
from dset import idx2char
# We use cross entropy loss
loss_func = nn.CrossEntropyLoss(reduction='mean')


def compute_loss(rnn, xNy, h_list, device):
    """
    compute_loss for a given RNN model using loss_func

    Args:
        RNN: model to be trained
        xNy (tuple): the input and target pair of the form (input, target)
        h_list (list): list of hidden states. Each hidden state is a torch.tensor
        device(str): 'cpu' or 'cuda'

    Returns:
        torch.tensor: value of the loss
    """
    x_t, y_t = xNy
    x_t=x_t.to(device, non_blocking=True)
    y_t=y_t.to(device, non_blocking=True)
    loss = 0.
    for i in range(x_t.shape[1]):
        out, h_list = rnn(x_t[:, i, :], h_list)
        loss += loss_func(out, y_t[:, i])
    return loss


def print_function(max_len, rnn, char_size, h_size, depth, mode):
    """ Generate text and print it using rnn.
    Args: 
        max_len (int): maximum length of generated text
        rnn: RNN model    
        char_size: number of characters in the vocabulary.
        h_size: size of hidden layer.
        mode (str): one of "RNN", "LSTM", "GRU"
    """
    rnn.eval()
    seed = torch.zeros((1, char_size))
    seed[0, rand.randint(0, char_size)] = 1
    if mode == "RNN" or mode == "GRU":
        h = [torch.zeros((1, h_size)) for i in range(depth)]
    elif mode == "LSTM":
        h = ([torch.zeros((1, h_size)) for i in range(depth)], [torch.zeros((1, h_size)) for i in range(depth)])
    generated = []
    out_text = ''
    with torch.no_grad():
        for i in range(max_len):
            out, h = rnn(seed, h)
            p = torch.nn.functional.softmax(out, dim=1)
            p = np.array(p)
            max_idx = np.random.choice(range(char_size), p=p.ravel())
            char = idx2char[max_idx.item()]
            out_text += char
            seed = torch.zeros((1, char_size))
            seed[0, max_idx] = 1
    print(out_text)
