import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, x_size, y_size, h_size, depth=1, dropout=0.0):
        """
        Implementation of basic RNN.

        Args:
            x_size (int): size of input
            y_size (int): size of output
            h_size (int): size of hidden state
            depth (int): number of hidden layers
            dropout (float): dropout probability, 0 <= dropout < 1
        """
        super().__init__()

        self.W_horizontal_list = nn.ModuleList()  # list of linear layers in 'time' direction.
        for i in range(depth):
            self.W_horizontal_list.append(nn.Linear(h_size, h_size))

        self.W_vertical_list = nn.ModuleList()  # list of linear layers in 'depth' direction, excluding the output layer.
        self.W_xh = nn.Linear(x_size, h_size, bias=False)
        self.W_vertical_list.append(self.W_xh)
        for i in range(depth - 1):
            self.W_vertical_list.append(nn.Linear(h_size, h_size))

        assert dropout >= 0.0 and dropout < 1.0, "dropout probability should be less than 1.0 and greater than or equal to 0.0"
        self.dropout_list = nn.ModuleList()  # list of dropout layers
        for i in range(depth):
            self.dropout_list.append(nn.Dropout(dropout))

        self.W_hy = nn.Linear(h_size, y_size)  # the output layer.

        self._init_weights()  # initialize weights and biases of linear layers.

    def forward(self, x, h_list):
        h_next_list = []  # list of hidden states for next 'time'.
        x_in = x
        for (h, horizontal, vertical, dropout_layer) in zip(h_list, self.W_horizontal_list, self.W_vertical_list, self.dropout_list):
            h_next = torch.tanh(vertical(x_in) + horizontal(h))
            h_next_list.append(h_next)
            x_in = dropout_layer(h_next)  # input to next layer; insert dropout before next layer
        y = self.W_hy(x_in)  # output
        return y, h_next_list

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear
            }:
                nn.init.uniform_(m.weight.data, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.uniform_(m.bias.data, -0.0001, 0.0001)


class LSTM(nn.Module):
    def __init__(self, x_size, y_size, h_size, depth=1, dropout=0.):
        """
        Implementation of LSTM.

        Args:
            x_size (int): size of input
            y_size (int): size of output
            h_size (int): size of hidden state (which is equivalent to the size of cell state)
            depth (int): number of hidden layers
            dropout (float): dropout probability, 0 <= dropout < 1
        """
        super().__init__()

        self.W_horizontal_list_f = nn.ModuleList()  # list of linear layers in 'time' direction for forget gate
        for i in range(depth):
            self.W_horizontal_list_f.append(nn.Linear(h_size, h_size))

        self.W_horizontal_list_i = nn.ModuleList()  # list of linear layers in 'time' direction for input gate
        for i in range(depth):
            self.W_horizontal_list_i.append(nn.Linear(h_size, h_size))

        self.W_horizontal_list_o = nn.ModuleList()  # list of linear layers in 'time' direction for output gate
        for i in range(depth):
            self.W_horizontal_list_o.append(nn.Linear(h_size, h_size))

        self.W_horizontal_list_c = nn.ModuleList()  # list of linear layers in 'time' direction for new cell state content
        for i in range(depth):
            self.W_horizontal_list_c.append(nn.Linear(h_size, h_size))

        self.W_vertical_list_f = nn.ModuleList()  # list of linear layers in 'depth' direction for forget gate
        self.W_xf = nn.Linear(x_size, h_size, bias=False)
        self.W_vertical_list_f.append(self.W_xf)
        for i in range(depth - 1):
            self.W_vertical_list_f.append(nn.Linear(h_size, h_size))

        self.W_vertical_list_i = nn.ModuleList()  # list of linear layers in 'depth' direction for input gate
        self.W_xi = nn.Linear(x_size, h_size, bias=False)
        self.W_vertical_list_i.append(self.W_xi)
        for i in range(depth - 1):
            self.W_vertical_list_i.append(nn.Linear(h_size, h_size))

        self.W_vertical_list_o = nn.ModuleList()  # list of linear layers in 'depth' direction for output gate
        self.W_xo = nn.Linear(x_size, h_size, bias=False)
        self.W_vertical_list_o.append(self.W_xo)
        for i in range(depth - 1):
            self.W_vertical_list_o.append(nn.Linear(h_size, h_size))

        self.W_vertical_list_c = nn.ModuleList()  # list of linear layers in 'depth' direction for new cell state content
        self.W_xc = nn.Linear(x_size, h_size, bias=False)
        self.W_vertical_list_c.append(self.W_xc)
        for i in range(depth - 1):
            self.W_vertical_list_c.append(nn.Linear(h_size, h_size))

        assert dropout >= 0.0 and dropout < 1.0, "dropout probability should be less than 1.0 and greater than or equal to 0.0"
        self.dropout_list = nn.ModuleList()  # list of dropout layers
        for i in range(depth):
            self.dropout_list.append(nn.Dropout(dropout))

        self.W_hy = nn.Linear(h_size, y_size)  # the output layer.

        self._init_weights()  # initialize weights and biases of linear layers.

    def forward(self, x, hc_list):
        h_next_list = []
        c_next_list = []
        h_list, c_list = hc_list
        x_in = x
        for (h,
             c,
             horizontal_f,
             horizontal_i,
             horizontal_o,
             horizontal_c,
             vertical_f,
             vertical_i,
             vertical_o,
             vertical_c,
             dropout_layer,
             ) in zip(h_list,
                      c_list,
                      self.W_horizontal_list_f,
                      self.W_horizontal_list_i,
                      self.W_horizontal_list_o,
                      self.W_horizontal_list_c,
                      self.W_vertical_list_f,
                      self.W_vertical_list_i,
                      self.W_vertical_list_o,
                      self.W_vertical_list_c,
                      self.dropout_list,
                      ):
            f_next = torch.sigmoid(horizontal_f(h) + vertical_f(x_in))
            i_next = torch.sigmoid(horizontal_i(h) + vertical_i(x_in))
            o_next = torch.sigmoid(horizontal_o(h) + vertical_o(x_in))
            c_new = torch.sigmoid(horizontal_c(h) + vertical_c(x_in))

            c_next = f_next * c + i_next * c_new
            h_next = o_next * torch.tanh(c_next)

            h_next_list.append(h_next)
            c_next_list.append(c_next)

            x_in = dropout_layer(h_next)  # input to next layer; insert dropout before next layer

        y = self.W_hy(x_in)
        return y, (h_next_list, c_next_list)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear
            }:
                nn.init.uniform_(m.weight.data, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.uniform_(m.bias.data, -0.0001, 0.0001)
        for m in self.W_horizontal_list_f:
            nn.init.uniform_(m.bias.data, 0.9999, 1.0001)


class GRU(nn.Module):
    def __init__(self, x_size, y_size, h_size, depth=1, dropout=0.):
        """
        Implementation of GRU.

        Args:
            x_size (int): size of input
            y_size (int): size of output
            h_size (int): size of hidden state (which is equivalent to the size of cell state)
            depth (int): number of hidden layers
            dropout (float): dropout probability, 0 <= dropout < 1
        """
        super().__init__()

        self.W_horizontal_list_u = nn.ModuleList()  # list of linear layers in 'time' direction for update gate
        for i in range(depth):
            self.W_horizontal_list_u.append(nn.Linear(h_size, h_size))

        self.W_horizontal_list_r = nn.ModuleList()  # list of linear layers in 'time' direction for reset gate
        for i in range(depth):
            self.W_horizontal_list_r.append(nn.Linear(h_size, h_size))

        self.W_horizontal_list_h = nn.ModuleList()  # list of linear layers in 'time' direction for new hidden state content
        for i in range(depth):
            self.W_horizontal_list_h.append(nn.Linear(h_size, h_size))

        self.W_vertical_list_u = nn.ModuleList()  # list of linear layers in 'depth' direction for update gate
        self.W_xu = nn.Linear(x_size, h_size, bias=False)
        self.W_vertical_list_u.append(self.W_xu)
        for i in range(depth - 1):
            self.W_vertical_list_u.append(nn.Linear(h_size, h_size))

        self.W_vertical_list_r = nn.ModuleList()  # list of linear layers in 'depth' direction for reset gate
        self.W_xr = nn.Linear(x_size, h_size, bias=False)
        self.W_vertical_list_r.append(self.W_xr)
        for i in range(depth - 1):
            self.W_vertical_list_r.append(nn.Linear(h_size, h_size))

        self.W_vertical_list_h = nn.ModuleList()  # list of linear layers in 'depth' direction for new hidden state content
        self.W_xh = nn.Linear(x_size, h_size, bias=False)
        self.W_vertical_list_h.append(self.W_xh)
        for i in range(depth - 1):
            self.W_vertical_list_h.append(nn.Linear(h_size, h_size))

        assert dropout >= 0.0 and dropout < 1.0, "dropout probability should be less than 1.0 and greater than or equal to 0.0"
        self.dropout_list = nn.ModuleList()  # list of dropout layers
        for i in range(depth):
            self.dropout_list.append(nn.Dropout(dropout))

        self.W_hy = nn.Linear(h_size, y_size)  # the output layer.

        self._init_weights()  # initialize weights and biases of linear layers.

    def forward(self, x, h_list):
        h_next_list = []
        h_list
        x_in = x

        for (h,
             horizontal_u,
             horizontal_r,
             horizontal_h,
             vertical_u,
             vertical_r,
             vertical_h,
             dropout_layer,
             ) in zip(h_list,
                      self.W_horizontal_list_u,
                      self.W_horizontal_list_r,
                      self.W_horizontal_list_h,
                      self.W_vertical_list_u,
                      self.W_vertical_list_r,
                      self.W_vertical_list_h,
                      self.dropout_list,
                      ):
            u_next = torch.sigmoid(horizontal_u(h) + vertical_u(x_in))
            r_next = torch.sigmoid(horizontal_r(h) + vertical_r(x_in))
            h_new = torch.tanh(horizontal_h(r_next * h) + vertical_h(x_in))

            h_next = (1. - u_next) * h + u_next * h_new

            h_next_list.append(h_next)

            x_in = dropout_layer(h_next)  # input to next layer; insert dropout before next layer

        y = self.W_hy(x_in)
        return y, h_next_list

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear
            }:
                nn.init.uniform_(m.weight.data, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.uniform_(m.bias.data, -0.0001, 0.0001)
        for m in self.W_horizontal_list_r:
            nn.init.uniform_(m.bias.data, -1.0001, -0.9999)
