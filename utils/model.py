import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pandas as pd
from copy import deepcopy


########################################################################################################################
#                                                        DYconv                                                        #
########################################################################################################################
class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, freq_size, kernel_size, stride=1, padding=0, groups=1, bias=False,
                 n_basis_kernels=4, temperature=31, reduction=4, pool_dim='freq', conv1d_kernel=[3, 1],
                 dilated_DY=0, dilation_size=[[0, 0], [0, 0], [0, 0], [0, 0]], dy_chan_proportion=None, aggconv=False):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_basis_kernels = n_basis_kernels
        self.pool_dim = pool_dim
        self.groups = groups
        self.dilated_DY = dilated_DY
        self.dilation_size = dilation_size
        self.dy_chan_proportion = dy_chan_proportion
        self.aggconv = aggconv

        if dy_chan_proportion is not None:
            self.n_attention = len(dilation_size)
            dy_out_planes = int(out_planes * dy_chan_proportion[0] / dy_chan_proportion[1])
            self.dy_out_planes = dy_out_planes
            self.stt_out_plane = out_planes - dy_out_planes * self.n_attention

            if not self.dilated_DY:
                self.dilation_size = []
                for _ in range(self.n_attention):
                    if self.n_basis_kernels == 4:
                        self.dilation_size.append([[1, 1], [1, 1], [1, 1], [1, 1]])

            if not aggconv:
                if self.stt_out_plane > 0:
                    self.stt_conv = nn.Conv2d(in_planes, self.stt_out_plane, kernel_size, stride, padding, bias=bias)

                self.weight = []
                for n_bk in n_basis_kernels:
                    self.weight.append(nn.Parameter(torch.randn(n_bk, dy_out_planes, in_planes,
                                                                self.kernel_size, self.kernel_size)),
                                                    requires_grad=True)
                for j in range(self.n_attention):
                    for i in range(self.n_basis_kernels):
                        nn.init.kaiming_normal_(self.weight[j, i])

                self.bias = []
                if bias:
                    self.bias.append(nn.Parameter(torch.Tensor(self.n_attention, n_basis_kernels, dy_out_planes),
                                                  requires_grad=True))
                else:
                    self.bias = None

            else:
                output_sizes = [0, 0, 0]
                for i in range(self.n_attention):
                    for dil in self.dilation_size[i]:
                        output_sizes[dil[1]-1] += 1
                self.output_sizes = output_sizes

                self.conv_dil1 = nn.Conv2d(in_planes,
                                           self.stt_out_plane + dy_out_planes * output_sizes[0], kernel_size, stride,
                                           self.padding, bias=bias)

                if self.output_sizes[1] > 0:
                    self.conv_dil2 = nn.Conv2d(in_planes, dy_out_planes * output_sizes[1], kernel_size, stride,
                                               (self.padding + 1, self.padding + 1), dilation=2, bias=bias)

                if self.output_sizes[2] > 0:
                    self.conv_dil3 = nn.Conv2d(in_planes, dy_out_planes * output_sizes[2], kernel_size, stride,
                                               (self.padding + 2, self.padding + 2), dilation=3, bias=bias)

            self.attentions = []
            if isinstance(n_basis_kernels, int):
                n_basis_kernels = [n_basis_kernels] * self.n_attention
            for i in range(self.n_attention):
                if i == 0:
                    self.attention_0 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_0)

                elif i == 1:
                    self.attention_1 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_1)

                elif i == 2:
                    self.attention_2 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_2)

                elif i == 3:
                    self.attention_3 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_3)

                elif i == 4:
                    self.attention_4 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_4)

                elif i == 5:
                    self.attention_5 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_5)

                elif i == 6:
                    self.attention_6 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_6)

                elif i == 7:
                    self.attention_7 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_7)

                elif i == 8:
                    self.attention_8 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_8)

                elif i == 9:
                    self.attention_9 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_9)

                elif i == 10:
                    self.attention_10 = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_10)

                elif i == 11:
                    self.attention_11= attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                                   n_basis_kernels[i], temperature, reduction, pool_dim)
                    self.attentions.append(self.attention_11)


        else:
            self.n_attention = 1
            dy_out_planes = out_planes
            self.dy_out_planes = out_planes
            self.attention = attention2d(in_planes, conv1d_kernel, freq_size, self.stride,
                                         n_basis_kernels, temperature, reduction, pool_dim)

            self.weight = nn.Parameter(torch.randn(n_basis_kernels, dy_out_planes, in_planes,
                                                   self.kernel_size, self.kernel_size),
                                       requires_grad=True)
            for i in range(self.n_basis_kernels):
                nn.init.kaiming_normal_(self.weight[i])

            if bias:
                self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, dy_out_planes), requires_grad=True)
            else:
                self.bias = None



    def forward(self, x):                                           # x size : [bs, in_chan, frames, freqs]
        if self.dy_chan_proportion is not None:
            if not self.aggconv:
                if self.stt_out_plane:
                    stt_output = self.stt_conv(x)
                att_outputs = ()
                for i in range(self.n_attention):
                    bias = self.bias[i] if self.bias is not None else None
                    att_output = self.attention_forward(x, self.attentions[i], self.dilation_size[i], self.weight[i], bias)
                    att_outputs += (att_output,)
            else:
                out_dil1 = self.conv_dil1(x)
                if self.output_sizes[1] > 0:
                    out_dil2 = self.conv_dil2(x)
                if self.output_sizes[2] > 0:
                    out_dil3 = self.conv_dil3(x)
                if self.stt_out_plane > 0:
                    stt_output = out_dil1[:, :self.stt_out_plane]

                conv_start_idxes = [self.stt_out_plane, 0, 0]
                att_outputs = ()
                for i in range(self.n_attention):
                    bk_outs = []
                    for dil_size in self.dilation_size[i]:
                        if dil_size[1] == 1:
                            bk_outs.append(out_dil1[:, conv_start_idxes[0]: conv_start_idxes[0] + self.dy_out_planes])
                            conv_start_idxes[0] += self.dy_out_planes
                        if dil_size[1] == 2:
                            bk_outs.append(out_dil2[:, conv_start_idxes[1]: conv_start_idxes[1] + self.dy_out_planes])
                            conv_start_idxes[1] += self.dy_out_planes
                        if dil_size[1] == 3:
                            bk_outs.append(out_dil3[:, conv_start_idxes[2]: conv_start_idxes[2] + self.dy_out_planes])
                            conv_start_idxes[2] += self.dy_out_planes
                    att_output = self.attention_forward_aggconv(x, self.attentions[i], bk_outs)
                    att_outputs += (att_output,)

            if self.stt_out_plane > 0:
                output = torch.cat((stt_output,) + att_outputs, dim=1)
            else:
                output = torch.cat(att_outputs, dim=1)
        else:
            output = self.attention_forward(x, self.attention, self.dilation_size[0], self.weight, self.bias)

        return output


    def attention_forward_aggconv(self, x, attention, bk_outs):
        kernel_attention = attention(x)  # kernel_attention size : [bs, n_ker, 1, 1, freqs]
        output = torch.stack(bk_outs, dim=1)

        if self.pool_dim in ['freq']:
            assert kernel_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim in ['time']:
            assert kernel_attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * kernel_attention, dim=1)  # output size : [bs, out_chan, frames, freqs]

        return output


    def attention_forward(self, x, attention, dilation_size, weight, bias):
        kernel_attention = attention(x)                # kernel_attention size : [bs, n_ker, 1, 1, freqs]
        if self.dilated_DY:
            output = []
            for i in range(self.n_basis_kernels):
                padding = (self.padding + dilation_size[i][0] - 1, self.padding + dilation_size[i][1] - 1)
                if bias is not None:
                    output.append(F.conv2d(x, weight=weight[i], bias=bias[i], stride=self.stride,
                                           padding=padding, dilation=dilation_size[i], groups=self.groups))
                else:
                    output.append(F.conv2d(x, weight=weight[i], bias=None, stride=self.stride,
                                           padding=padding, dilation=dilation_size[i], groups=self.groups))

            output = torch.stack(output, dim=1)

        else:
            aggregate_weight = weight.view(-1, self.in_planes, self.kernel_size, self.kernel_size)
                                                                        # weight size : [n_ker * out_chan, in_chan, ks, ks]

            if bias is not None:
                aggregate_bias = bias.view(-1)
                output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                                  groups=self.groups)
            else:
                output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                                  groups=self.groups)
                                                                        # output size : [bs, n_ker * out_chan, frames, freqs]

            output = output.view(x.size(0), self.n_basis_kernels, self.dy_out_planes, output.size(-2), output.size(-1))
                                                                       # output size : [bs, n_ker, out_chan, frames, freqs]

        if self.pool_dim in ['freq']:
            assert kernel_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim in ['time']:
            assert kernel_attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * kernel_attention, dim=1)  # output size : [bs, out_chan, frames, freqs]

        return output


class attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, freq_size, stride, n_basis_kernels,
                 temperature, reduction, pool_dim):
        super(attention2d, self).__init__()
        self.freq_size = freq_size
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = in_planes // reduction
        if hidden_planes < 4:
            hidden_planes = 4

        padding_1 = int((kernel_size[0] - 1) / 2)
        padding_2 = int((kernel_size[1] - 1) / 2)
        if pool_dim == 'both':
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)
        else:
            self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size[0], stride=stride, padding=padding_1,
                                     bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, kernel_size[1], padding=padding_2, bias=True)


        # initialize
        if pool_dim in ["freq", "time"]:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):                                            # x size : [bs, chan, frames, freqs]
        ### Pool dimensions and apply pre-processings
        if self.pool_dim == 'freq':                               #TDY
            x = torch.mean(x, dim=3)                                 # x size : [bs, chan, frames]
        elif self.pool_dim == 'time': #FDY
            x = torch.mean(x, dim=2)                             # x size : [bs, chan, freqs]
        elif self.pool_dim == 'both':                           #DY
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

        ### extract attention weights
        if self.pool_dim == 'both':
            x = self.relu(self.fc1(x))                                             #x size : [bs, sqzd_chan]
            att = self.fc2(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)            #att size : [bs, n_ker, 1, 1, 1]
        elif self.pool_dim == 'freq':
            x = self.relu(self.bn(self.conv1d1(x)))                                #x size : [bs, sqzd_chan, frames]
            att = self.conv1d2(x).unsqueeze(2).unsqueeze(4)                        #x size : [bs, n_ker, 1, frames, 1]
        else:  #self.pool_dim == 'time', FDY
            x = self.relu(self.bn(self.conv1d1(x)))                                #x size : [bs, sqzd_chan, freqs]
            att = self.conv1d2(x)                                                  #att size : [bs, n_ker, freqs]
            att = att.unsqueeze(2).unsqueeze(3)                                    #att size : [bs, n_ker, 1, 1, freqs]

        return F.softmax(att / self.temperature, 1)


########################################################################################################################
#                                                        CRNN                                                          #
########################################################################################################################
class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        # ores = x * sig
        return res


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        #self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x

class DYCNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 dilation=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 pre_conv=None,
                 normalization="batch",
                 DY_layers=[0, 0, 0, 0, 0, 0, 0],
                 n_basis_kernels=4,
                 temperature=31,
                 dy_reduction=4,
                 pool_dim='freq',
                 conv1d_kernel=[3, 1],
                 dilated_DY=[0, 0, 0, 0, 0, 0, 0],
                 dilation_size=[[0, 0], [0, 0], [0, 0], [0, 0]],
                 dy_chan_proportion=None,
                 aggconv=False,):
        super(DYCNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()
        if len(n_filt) == 7:
            freq_dims = [128, 64, 32, 16, 8, 4, 2]

        if pre_conv is not None:
            cnn.add_module("pre_conv", nn.Conv2d(n_input_ch, pre_conv, 3, 1, 1))
            n_input_ch = pre_conv

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
            # convolution
            if DY_layers[i] == 1:
                cnn.add_module("conv{0}".format(i), Dynamic_conv2d(in_dim, out_dim, freq_dims[i], kernel[i], stride[i],
                                                                   pad[i],
                                                                   n_basis_kernels=n_basis_kernels,
                                                                   temperature=temperature,
                                                                   pool_dim=pool_dim,
                                                                   reduction=dy_reduction,
                                                                   conv1d_kernel=conv1d_kernel,
                                                                   dilated_DY=dilated_DY[i],
                                                                   dilation_size=dilation_size,
                                                                   dy_chan_proportion=dy_chan_proportion,
                                                                   aggconv=aggconv))
            else:
                cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i],
                                                              dilation[i]))


            # normalization
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            # non-linearity
            if activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))

            # dropout
            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):    #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class DYCRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 n_class=10,
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 conv_dropout=0.5,
                 **convkwargs):
        super(DYCRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.cnn = DYCNN(n_input_ch=n_input_ch, dropout=conv_dropout, **convkwargs)

        rnn_in = self.cnn.n_filt[-1]
        self.rnn = BiGRU(n_in=rnn_in, n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()

        linear_in = n_RNN_cell * 2
        self.linear = nn.Linear(linear_in, n_class)
        if self.attention:
            self.linear_att = nn.Linear(linear_in, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(dim=1)  # softmax on time dimension
            elif self.attention == "class":
                self.softmax = nn.Softmax(dim=-1)                            # softmax on class dimension


    def forward(self, x):                                                # input size: [bs, freqs, frames]
        #cnn
        x = x.transpose(1, 2).unsqueeze(1)                               # x size: [bs, chan, frames, freqs]
        x = self.cnn(x)                                                  # x size: [bs, chan, frames, 1]
        x = x.squeeze(-1)                                                # x size: [bs, chan, frames]
        x = x.permute(0, 2, 1)                                           # x size: [bs, frames, chan]

        #rnn
        x = self.rnn(x)                                                  # x size: [bs, frames, 2 * chan]
        x = self.dropout(x)
        strong = self.linear(x)                                          # strong size: [bs, frames, n_class]
        strong = self.sigmoid(strong)
        if self.attention:
            attention = self.linear_att(x)                               # attention size: [bs, frames, n_class]
            attention = self.softmax(attention)                          # attention size: [bs, frames, n_class]
            attention = torch.clamp(attention, min=1e-7, max=1)
            weak = (strong * attention).sum(1) / attention.sum(1)        # weak size: [bs, n_class]
        else:
            weak = strong.mean(1)

        return strong.transpose(1, 2), weak
