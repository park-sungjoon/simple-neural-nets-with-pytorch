# simple-neural-nets-with-pytorch

Implementation of some neural networks with pytorch.

## Character-level language model
Based on [Andrej Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [Stanford NLP lecture (cs224N)](http://web.stanford.edu/class/cs224n/).
Here, I implement vanilla recurrent neural network (RNN), long short-term memory (LSTM) and gated recurrent unit (GRU) to generate random text based on `input.text`.
In LSTM, the forget gate bias is initialized to approximately 1, and in GRU, the bias for reset gate is initialized to approximately -1, see this [page](https://danijar.com/tips-for-training-recurrent-neural-networks/).
