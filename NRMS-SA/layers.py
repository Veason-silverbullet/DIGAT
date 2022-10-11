import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D(nn.Module):
    def __init__(self, cnn_method: str, in_channels: int, cnn_kernel_num: int, cnn_window_size: int):
        super(Conv1D, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group5']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        if self.cnn_method == 'naive':
            self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num, kernel_size=cnn_window_size, padding=(cnn_window_size - 1) // 2)
        elif self.cnn_method == 'group3':
            assert cnn_kernel_num % 3 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=5, padding=2)
        else:
            assert cnn_kernel_num % 5 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=2, padding=0)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=4, padding=1)
            self.conv5 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=5, padding=2)
        self.device = torch.device('cuda')

    def initialize(self):
        pass

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, cnn_kernel_num, length]
    def forward(self, feature):
        if self.cnn_method == 'naive':
            return F.relu(self.conv(feature))
        elif self.cnn_method == 'group3':
            return F.relu(torch.cat([self.conv1(feature), self.conv2(feature), self.conv3(feature)], dim=1))
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1], device=self.device)
            return F.relu(torch.cat([self.conv1(feature), \
                                     self.conv2(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv3(feature), \
                                     self.conv4(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv5(feature)], dim=1))


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.len_q = len_q
        self.len_k = len_k
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = self.h * self.d_v
        self.attention_scalar = math.sqrt(float(self.d_k))
        self.W_K = nn.Linear(in_features=d_model, out_features=self.h*self.d_k, bias=False)
        self.W_Q = nn.Linear(in_features=d_model, out_features=self.h*self.d_k, bias=True)
        self.W_V = nn.Linear(in_features=d_model, out_features=self.h*self.d_v, bias=True)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)

    # Input
    # Q    : [batch_size, len_q, d_model]
    # K    : [batch_size, len_k, d_model]
    # V    : [batch_size, len_k, d_model]
    # mask : [batch_size, len_k]
    # Output
    # out  : [batch_size, len_q, h * d_v]
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view([batch_size, self.len_q, self.h, self.d_k])                                           # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, self.len_k, self.h, self.d_k])                                           # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, self.len_k, self.h, self.d_v])                                           # [batch_size, len_k, h, d_v]
        Q = Q.transpose(1, 2).contiguous().view([batch_size * self.h, self.len_q, self.d_k])                       # [batch_size * h, len_q, d_k]
        K = K.transpose(1, 2).contiguous().view([batch_size * self.h, self.len_k, self.d_k])                       # [batch_size * h, len_k, d_k]
        V = V.transpose(1, 2).contiguous().view([batch_size * self.h, self.len_k, self.d_v])                       # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.transpose(1, 2).contiguous()) / self.attention_scalar                                   # [batch_size * h, len_q, len_k]
        if mask != None:
            _mask = mask.repeat([1, self.h]).view([batch_size * self.h, 1, self.len_k]).repeat([1, self.len_q, 1]) # [batch_size * h, len_q, len_k]
            alpha = F.softmax(A.masked_fill(_mask == 0, -1e9), dim=2)                                              # [batch_size * h, len_q, len_k]
        else:
            alpha = F.softmax(A, dim=2)                                                                            # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])                                 # [batch_size, h, len_q, d_v]
        out = out.transpose(1, 2).contiguous().view([batch_size, self.len_q, self.out_dim])                        # [batch_size, len_q, h * d_v]
        return out


class Attention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(Attention, self).__init__()
        self.affine1 = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.affine2 = nn.Linear(in_features=attention_dim, out_features=1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, mask=None):
        attention = torch.tanh(self.affine1(feature))                                 # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)                                    # [batch_size, length]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1) # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                              # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                # [batch_size, feature_dim]
        return out


class CandidateAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(CandidateAttention, self).__init__()
        self.feature_affine = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=False)
        self.query_affine = nn.Linear(in_features=query_dim, out_features=attention_dim, bias=True)
        self.attention_affine = nn.Linear(in_features=attention_dim, out_features=1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.feature_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.query_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.query_affine.bias)
        nn.init.xavier_uniform_(self.attention_affine.weight)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = self.attention_affine(
            torch.tanh(self.feature_affine(feature) + self.query_affine(query).unsqueeze(dim=1))
        ).squeeze(dim=2)                                                # [batch_size, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)    # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                 # [batch_size, feature_num]
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1) # [batch_size, feature_dim]
        return out


class MultipleCandidateAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(MultipleCandidateAttention, self).__init__()
        self.feature_affine = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=False)
        self.query_affine = nn.Linear(in_features=query_dim, out_features=attention_dim, bias=True)
        self.attention_affine = nn.Linear(in_features=attention_dim, out_features=1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.feature_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.query_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.query_affine.bias)
        nn.init.xavier_uniform_(self.attention_affine.weight)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_num, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, query_num, feature_dim]
    def forward(self, feature, query, mask=None):
        query_num = query.size(1)
        a = self.attention_affine(
                torch.tanh(self.feature_affine(feature).unsqueeze(dim=1) + self.query_affine(query).unsqueeze(dim=2))
            ).squeeze(dim=3)                                                                                    # [batch_size, query_num, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask.unsqueeze(dim=1).expand(-1, query_num, -1) == 0, -1e9), dim=2) # [batch_size, query_num, feature_num]
        else:
            alpha = F.softmax(a, dim=2)                                                                         # [batch_size, query_num, feature_num]
        out = torch.bmm(alpha, feature)                                                                         # [batch_size, query_num, feature_dim]
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.K = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=False)
        self.Q = nn.Linear(in_features=query_dim, out_features=attention_dim, bias=True)
        self.attention_scalar = math.sqrt(float(attention_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = torch.bmm(self.K(feature), self.Q(query).unsqueeze(dim=2)).squeeze(dim=2) / self.attention_scalar # [batch_size, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)                                          # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                                                       # [batch_size, feature_num]
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1)                                       # [batch_size, feature_dim]
        return out


class MultipleScaledDotProductAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(MultipleScaledDotProductAttention, self).__init__()
        self.K = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.Q = nn.Linear(in_features=query_dim, out_features=attention_dim, bias=True)
        self.attention_scalar = math.sqrt(float(attention_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.zeros_(self.K.bias)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_num, query_dim]
    # mask    : [batch_size, query_num, feature_num]
    # Output
    # out     : [batch_size, query_num, feature_dim]
    def forward(self, feature, query, mask=None):
        a = torch.bmm(self.Q(query), self.K(feature).transpose(1, 2)) / self.attention_scalar # [batch_size, query_num, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=2)                          # [batch_size, query_num, feature_num]
        else:
            alpha = F.softmax(a, dim=2)                                                       # [batch_size, query_num, feature_num]
        out = torch.bmm(alpha, feature)                                                       # [batch_size, query_num, feature_dim]
        return out


class DualScaledDotProductAttention(nn.Module):
    def __init__(self, feature_dim1: int, feature_dim2: int, attention_dim: int):
        super(DualScaledDotProductAttention, self).__init__()
        self.f1 = nn.Linear(in_features=feature_dim1, out_features=attention_dim, bias=True)
        self.f2 = nn.Linear(in_features=feature_dim2, out_features=attention_dim, bias=True)
        self.attention_scalar = math.sqrt(float(attention_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.zeros_(self.f1.bias)
        nn.init.xavier_uniform_(self.f2.weight)
        nn.init.zeros_(self.f2.bias)

    # Input
    # feature1 : [batch_size, feature_num1, feature_dim1]
    # feature2 : [batch_size, feature_num2, feature_dim2]
    # mask     : [batch_size, feature_num1, feature_num2]
    # Output
    # out1     : [batch_size, feature_num1, feature_dim2]
    # out2     : [batch_size, feature_num2, feature_dim1]
    def forward(self, feature1, feature2, mask=None):
        a = torch.bmm(self.f1(feature1), self.f2(feature2).transpose(1, 2)) / self.attention_scalar # [batch_size, feature_num1, feature_num2]
        if mask is not None:
            masked_a = a.masked_fill(mask == 0, -1e9)                                               # [batch_size, feature_num1, feature_num2]
            alpha1 = F.softmax(masked_a, dim=2)                                                     # [batch_size, feature_num1, feature_num2]
            alpha2 = F.softmax(masked_a, dim=1)                                                     # [batch_size, feature_num1, feature_num2]
        else:
            alpha1 = F.softmax(a, dim=2)                                                            # [batch_size, feature_num1, feature_num2]
            alpha2 = F.softmax(a, dim=1)                                                            # [batch_size, feature_num1, feature_num2]
        out1 = torch.bmm(alpha2.transpose(1, 2), feature1)                                          # [batch_size, feature_num2, feature_dim1]
        out2 = torch.bmm(alpha1, feature2)                                                          # [batch_size, feature_num1, feature_dim2]
        return out1, out2


class DualScaledDotProductAttention_(nn.Module):
    def __init__(self, feature_dim):
        super(DualScaledDotProductAttention_, self).__init__()
        self.attention_scalar = math.sqrt(float(feature_dim))

    def initialize(self):
        pass

    # Input
    # feature1 : [batch_size, feature_num1, feature_dim]
    # feature2 : [batch_size, feature_num2, feature_dim]
    # mask     : [batch_size, feature_num1, feature_dim]
    # Output
    # out1     : [batch_size, feature_num1, feature_dim]
    # out2     : [batch_size, feature_num2, feature_dim]
    def forward(self, feature1, feature2, mask=None):
        a = torch.bmm(feature1, feature2.transpose(1, 2)) / self.attention_scalar # [batch_size, feature_num1, feature_num2]
        if mask is not None:
            masked_a = a.masked_fill(mask == 0, -1e9)                             # [batch_size, feature_num1, feature_num2]
            alpha1 = F.softmax(masked_a, dim=2)                                   # [batch_size, feature_num1, feature_num2]
            alpha2 = F.softmax(masked_a, dim=1)                                   # [batch_size, feature_num1, feature_num2]
        else:
            alpha1 = F.softmax(a, dim=2)                                          # [batch_size, feature_num1, feature_num2]
            alpha2 = F.softmax(a, dim=1)                                          # [batch_size, feature_num1, feature_num2]
        out1 = torch.bmm(alpha2.transpose(1, 2), feature1)                        # [batch_size, feature_num2, feature_dim]
        out2 = torch.bmm(alpha1, feature2)                                        # [batch_size, feature_num1, feature_dim]
        return out1, out2


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, residual=False, layer_norm=False):
        super(GCNLayer, self).__init__()
        self.residual = residual
        self.layer_norm = layer_norm
        if self.residual and in_dim != out_dim:
            raise Exception('To facilitate residual connection, in_dim must equal to out_dim')
        self.W = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        if self.layer_norm:
            self.layer_normalization = nn.LayerNorm(normalized_shape=[out_dim])

    def initialize(self):
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W.bias)

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = self.W(torch.bmm(graph, feature))
        if self.layer_norm:
            out = self.layer_normalization(out)
        out = F.relu(out)
        if self.residual:
            out += feature
        return out


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=0, num_layers=1, dropout=0.1, residual=False, layer_norm=False):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = []
        if self.num_layers == 1:
            self.gcn_layers.append(GCNLayer(in_dim, out_dim, residual=residual, layer_norm=layer_norm))
        else:
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.gcn_layers.append(GCNLayer(in_dim, hidden_dim, residual=residual, layer_norm=layer_norm))
            for i in range(1, self.num_layers - 1):
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, residual=residual, layer_norm=layer_norm))
            self.gcn_layers.append(GCNLayer(hidden_dim, out_dim, residual=residual, layer_norm=layer_norm))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

    def initialize(self):
        for gcn_layer in self.gcn_layers:
            gcn_layer.initialize()

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = feature
        for i in range(self.num_layers - 1):
            out = self.dropout(self.gcn_layers[i](out, graph))
        out = self.gcn_layers[self.num_layers - 1](out, graph)
        return out


class GatedRGCNLayer(nn.Module):
    def __init__(self, feature_dim):
        super(GatedRGCNLayer, self).__init__()
        self.fs = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        self.fr = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        self.fa = nn.Linear(in_features=feature_dim*2, out_features=feature_dim, bias=True)

    def initialize(self):
        nn.init.xavier_uniform_(self.fs.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.fs.bias)
        nn.init.xavier_uniform_(self.fr.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.fr.bias)
        nn.init.xavier_uniform_(self.fa.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.fa.bias)

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        _feature = self.fs(feature) + self.fr(torch.bmm(graph, feature))
        gate = torch.sigmoid(self.fa(torch.cat([_feature, feature], dim=2)))
        out = F.relu(_feature) * gate + feature * (1 - gate)
        return out


class GatedRGCN(nn.Module):
    def __init__(self, feature_dim, num_layers=1, dropout=0.1):
        super(GatedRGCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = [GatedRGCNLayer(feature_dim) for i in range(self.num_layers)]
        self.gcn_layers = nn.ModuleList(self.gcn_layers)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def initialize(self):
        for gcn_layer in self.gcn_layers:
            gcn_layer.initialize()

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = feature
        for i in range(self.num_layers - 1):
            out = self.dropout(self.gcn_layers[i](out, graph))
        out = self.gcn_layers[self.num_layers - 1](out, graph)
        return out


class GATLayer(nn.Module):
    def __init__(self, feature_dim, dropout=0.1, residual=False):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        self.Q = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        self.K = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        self.attention_scalar = math.sqrt(float(feature_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.residual = residual

    def initialize(self):
        pass

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        h = self.W(feature)                                                                          # [batch_size, node_num, feature_dim]
        e = self.leaky_relu(torch.bmm(self.Q(h), self.K(h).transpose(1, 2)) / self.attention_scalar) # [batch_size, node_num, node_num]
        a = self.dropout(F.softmax(e.masked_fill(graph == 0, -1e9), dim=2))                          # [batch_size, node_num, node_num]
        out = F.relu(torch.bmm(a, h))                                                                # [batch_size, node_num, feature_dim]
        if self.residual:
            out += feature
        return out


class GAT(nn.Module):
    def __init__(self, feature_dim, num_layers=1, dropout=0.1, residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.gat_layers = [GATLayer(feature_dim, dropout=dropout, residual=residual) for i in range(self.num_layers)]
        self.gat_layers = nn.ModuleList(self.gat_layers)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def initialize(self):
        for gat_layer in self.gat_layers:
            gat_layer.initialize()

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = feature
        for i in range(self.num_layers - 1):
            out = self.dropout(self.gat_layers[i](out, graph))
        out = self.gat_layers[self.num_layers - 1](out, graph)
        return out


class MultiheadGATLayer(nn.Module):
    def __init__(self, feature_dim, head_num, dropout=0.1, residual=False):
        super(MultiheadGATLayer, self).__init__()
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.V = nn.Linear(in_features=feature_dim, out_features=head_num*feature_dim, bias=True)
        self.Q = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        self.K = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        self.attention_scalar = math.sqrt(float(feature_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.residual = residual

    def initialize(self):
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.zeros_(self.K.bias)

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        batch_size = feature.size(0)
        node_num = feature.size(1)
        h = self.V(feature).view([batch_size, node_num, self.head_num, self.feature_dim]).transpose(1, 2).contiguous()      # [batch_size, head_num, node_num, feature_dim]
        Q = self.Q(h)                                                                                                       # [batch_size, head_num, node_num, feature_dim]
        K = self.K(h).permute(0, 1, 3, 2)                                                                                   # [batch_size, head_num, feature_dim, node_num]
        e = self.leaky_relu(torch.matmul(Q, K) / self.attention_scalar)                                                     # [batch_size, head_num, node_num, node_num]
        a = self.dropout(F.softmax(e.masked_fill(graph.unsqueeze(dim=1).repeat(1, self.head_num, 1, 1) == 0, -1e9), dim=3)) # [batch_size, head_num, node_num, node_num]
        out = F.relu(torch.matmul(a, h).mean(dim=1, keepdim=False))                                                         # [batch_size, node_num, feature_dim]
        if self.residual:
            out += feature
        return out


class MultiheadGAT(nn.Module):
    def __init__(self, feature_dim, head_num, num_layers=1, dropout=0.1, residual=False):
        super(MultiheadGAT, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.gat_layers = [MultiheadGATLayer(feature_dim, head_num, dropout=dropout, residual=residual) for i in range(self.num_layers)]
        self.gat_layers = nn.ModuleList(self.gat_layers)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def initialize(self):
        for gat_layer in self.gat_layers:
            gat_layer.initialize()

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = feature
        for i in range(self.num_layers - 1):
            out = self.dropout(self.gat_layers[i](out, graph))
        out = self.gat_layers[self.num_layers - 1](out, graph)
        return out
