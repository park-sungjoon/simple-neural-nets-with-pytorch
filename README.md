# simple-neural-nets-with-pytorch

Implementation of some neural networks with pytorch.

## Character-level language model
Based on [Andrej Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [Stanford NLP lecture (cs224N)](http://web.stanford.edu/class/cs224n/).
Here, I implement vanilla recurrent neural network (RNN), long short-term memory (LSTM) and gated recurrent unit (GRU) to generate random text based on `input.text`.
In LSTM, the forget gate bias is initialized to approximately 1, and in GRU, the bias for reset gate is initialized to approximately -1, see this [page](https://danijar.com/tips-for-training-recurrent-neural-networks/).

The RNN models are trained with batches of strings (having length 25) and dropout (0.5).
Sample text after 5000 iterations on Pride and Prejudice  (LSTM, depth=1, hidden state size = 512): 
>      Mr. Darcy believed, you know the whole windownce out as much and I have been proposals; for what I thought as before, seemed to listen a lively, for a lice him
>      only in Hertfordshire a gentlemen and the former concluded and Meryton Gard-nated
>      was
>      found herself he had one than she had been the
>      world. He is uncommon.



## Variational Autoencoder
Vanilla VAE on MNIST data based on Kingma and Welling's [paper]{https://arxiv.org/abs/1312.6114}.
<p align="center">
    <img src="vanilla VAE/reconstructed_[2, 4, 1, 6, 4, 7, 0, 9, 4, 8].pdf" width="400"\>
</p>
