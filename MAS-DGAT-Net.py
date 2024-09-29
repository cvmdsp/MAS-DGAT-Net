import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        # Define trainable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):

        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)
        # [B, N, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # [B, N, N, 1] => [B, N, N] Correlation coefficient of graph attention (unnormalized)

        zero_vec = -1e12 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):

        super(GAT, self).__init__()
        self.dropout = dropout

        # Define a multi head graph attention layer
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj1, adj2):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj1) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        # print(x.shape)
        x = F.elu(self.out_att(x, adj2))
        return F.log_softmax(x, dim=2)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=62, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(self.avg_pool(x).size())
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # print(out.size())
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print(avg_out)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # print('x.size', x.size())
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels=62, ratio=8):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(in_channels, ratio)
        self.SpatialAttention = SpatialAttention()

    def forward(self, x):
        x = self.ChannelAttention(x) * x
        x = self.SpatialAttention(x) * x
        return x


# Calculate Spearman correlation coefficient
class SpearmanCorrelation(nn.Module):  # Based on calculations between different electrodes, the output is [batch_size,channel,channel]
    def __init__(self):
        super(SpearmanCorrelation, self).__init__()
        self.rank = None
        self.weight = None

    def __call__(self, de_data):

        de_tensor = de_data.clone().detach()
        de_tensor = de_tensor.to(device)
        # de_tensor = torch.Tensor(de_data)
        batch_size, electrodes, freq = de_tensor.size()

        corr_tensor = torch.zeros(batch_size, electrodes, electrodes)

        for i in range(batch_size):
            if self.rank is None:
                self.rank = torch.empty(electrodes, dtype=torch.float32)
            if self.weight is None:
                self.weight = torch.zeros(electrodes, electrodes)

            for f in range(freq):

                if self.rank is None:
                    self.rank = torch.empty(electrodes, dtype=torch.float32)
                    for j in range(electrodes):
                        self.rank[j] = torch.argsort(de_tensor[i][j][f], descending=False).float()

                if self.weight is None:
                    self.weight = torch.zeros(electrodes, electrodes)
                    for j in range(electrodes):
                        for k in range(electrodes):
                            self.weight[j][k] = 1 / abs(self.rank[j] - self.rank[k] + 1)

            se = torch.sum(self.weight, dim=0, keepdim=True)
            corr = torch.sum(self.weight.unsqueeze(2) * self.weight.unsqueeze(1), dim=0) - se.t() * se / (
                        electrodes - 1)
            corr_tensor[i] += corr

        corr_tensor /= freq
        return corr_tensor


class senet(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(senet, self).__init__()
        '''-------------SE model-----------------------------'''
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, hidden_channel, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(hidden_channel, in_channel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.fc(self.gap(x).flatten(1, 2))
        x2 = torch.unsqueeze(x1, dim=2) * x
        return x2


class sperchannel(nn.Module):
    def __init__(self):
        super(sperchannel, self).__init__()
        self.conv1 = nn.Conv1d(62, 62, 3, padding=1)
        self.conv2 = nn.Conv1d(62, 62, 3, padding=1)
        self.conv3 = nn.Conv1d(62, 62, 3, padding=1)
        self.selu = nn.SELU()
        self.bn1 = nn.BatchNorm1d(62).to(device)
        self.bn2 = nn.BatchNorm1d(62).to(device)
        self.bn3 = nn.BatchNorm1d(62).to(device)
        self.senet1 = senet(62, 31)
        self.senet2 = senet(62, 31)

    def forward(self, x):
        x = x.to(device)
        x1 = self.bn1(self.selu(self.conv1(x)))
        x2 = self.senet1(x1)
        x3 = x2 + x
        x4 = self.bn2(self.selu(self.conv2(x3)))
        x5 = self.senet2(x4)
        x6 = x5 + x3
        return x6   # [64, 62, 58]


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):

        super(DepthWiseConv, self).__init__()
        # Channel wise convolution
        self.depth_conv = nn.Conv1d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=0,
                                    groups=in_channel)
        # Pointwise convolution
        self.point_conv = nn.Conv1d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class sperfre(nn.Module):
    def __init__(self):
        super(sperfre, self).__init__()
        self.conv1 = nn.Conv1d(5, 31, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(31)
        self.reconvs = nn.Conv1d(5, 31, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(31, 16, bias=False),  # c -> c/r
            nn.ReLU(),
            nn.Linear(16, 31, bias=False),  # c/r -> c
            nn.Sigmoid()
        )
        self.conv2 = nn.Conv1d(31, 62, 3, padding=1)
        self.reconv = nn.Conv1d(31, 62, 1)
        self.bn2 = nn.BatchNorm1d(62)
        self.selu = nn.SELU()

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.fc(self.gap(x1).flatten(1, 2))
        x3 = torch.unsqueeze(x2, dim=2) * x1 + self.reconvs(x)
        x4 = self.selu(x3)
        x5 = self.selu(self.bn2(self.conv2(x4))) + self.reconv(x4)
        return x5


class cabam_pro(nn.Module):
    def __init__(self):
        super(cabam_pro, self).__init__()
        self.cbam1 = CBAM()
        self.cbam2 = CBAM()
        self.conv1 = nn.Conv1d(62, 62, 3, padding=0)
        self.conv2 = nn.Conv1d(62, 62, 3, padding=0)
        self.selu = nn.SELU()
        self.bn1 = nn.BatchNorm1d(62)
        self.bn2 = nn.BatchNorm1d(62)

    def forward(self, x):
        x1 = self.selu(self.bn1(self.conv1(x)))
        x2 = torch.cat((x, x1), dim=2)
        x3 = self.cbam1(x2)
        x4 = self.selu(self.bn2(self.conv2(x3)))
        x5 = torch.cat((x1, x4), dim=2)
        x6 = self.cbam2(x5)
        return x6   # [64,62,9]


class deprocess(nn.Module):
    def __init__(self):
        super(deprocess, self).__init__()
        self.dwconv1 = DepthWiseConv(62, 62)
        self.dwconv2 = DepthWiseConv(62, 62)
        self.dwconv3 = DepthWiseConv(62, 62)
        self.selu = nn.SELU()
        self.bn1 = nn.BatchNorm1d(62)
        self.bn2 = nn.BatchNorm1d(62)
        self.bn3 = nn.BatchNorm1d(62)

    def forward(self, x):
        x1 = self.bn1(self.selu(self.dwconv1(x)))
        x2 = torch.cat((x, x1), dim=2)
        x3 = self.bn2(self.selu(self.dwconv2(x2)))
        x4 = torch.cat((x1, x3), dim=2)
        x5 = self.bn3(self.selu(self.dwconv3(x4)))   # 64,62,7
        return x5


class BLS(nn.Module):
    def __init__(self, in_nodes, feature_nodes, enhancement_nodes, out_nodes):
        super(BLS, self).__init__()
        self.fc1 = nn.Linear(in_nodes, feature_nodes)
        self.fc2 = nn.Linear(in_nodes, feature_nodes)
        self.fc3 = nn.Linear(in_nodes, feature_nodes)
        self.fc4 = nn.Linear(in_nodes, feature_nodes)
        self.fc5 = nn.Linear(in_nodes, feature_nodes)
        self.fc6 = nn.Linear(in_nodes, feature_nodes)
        self.fc7 = nn.Linear(in_nodes, feature_nodes)
        self.fc8 = nn.Linear(in_nodes, feature_nodes)
        self.fc9 = nn.Linear(in_nodes, feature_nodes)
        self.fc10 = nn.Linear(in_nodes, feature_nodes)

        self.fc31 = nn.Linear(feature_nodes * 10, enhancement_nodes)
        self.fc32 = nn.Linear(feature_nodes * 10 + enhancement_nodes, out_nodes)

    def forward(self, x):
        feature_nodes1 = torch.sigmoid(self.fc1(x))
        feature_nodes2 = torch.sigmoid(self.fc2(x))
        feature_nodes3 = torch.sigmoid(self.fc3(x))
        feature_nodes4 = torch.sigmoid(self.fc4(x))
        feature_nodes5 = torch.sigmoid(self.fc5(x))
        feature_nodes6 = torch.sigmoid(self.fc6(x))
        feature_nodes7 = torch.sigmoid(self.fc7(x))
        feature_nodes8 = torch.sigmoid(self.fc8(x))
        feature_nodes9 = torch.sigmoid(self.fc9(x))
        feature_nodes10 = torch.sigmoid(self.fc10(x))
        feature_nodes = torch.cat(
            [feature_nodes1, feature_nodes2, feature_nodes3, feature_nodes4, feature_nodes5, feature_nodes6,
             feature_nodes7, feature_nodes8, feature_nodes9, feature_nodes10], 1)
        enhancement_nodes = torch.sigmoid(self.fc31(feature_nodes))
        FeaAndEnhance = torch.cat([feature_nodes, enhancement_nodes], 1)
        result = self.fc32(FeaAndEnhance)  # [64,64]
        return result


def normalize_A(A, symmetry=False):

    # Symmetric normalization is applied to the adjacency matrix to convert it
    # into an effective similarity matrix, facilitating subsequent graph convolution operations。
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


class DGCNN_DE(nn.Module):  # input_format [batch_size, channel, frequency]
    def __init__(self, xdim, k_adj, num_out1, feature_nodes=300, enhancement_nodes=100, nclass=3):
        super(DGCNN_DE, self).__init__()
        self.K = k_adj
        self.BN1 = nn.BatchNorm1d(5)
        self.fc33 = nn.Linear(1458, 512)
        self.fc34 = nn.Linear(512, 128)
        self.fc35 = nn.Linear(128, nclass)
        self.dropout = nn.Dropout(p=0.1)
        self.sperman = SpearmanCorrelation()
        self.sperchannel = sperchannel()
        self.deprocess = deprocess()
        self.cbam_pro = cabam_pro()
        self.bls1 = BLS(3968, 200, 100, 1024)
        self.adj1 = nn.Parameter(torch.empty(62, 62))
        nn.init.xavier_normal_(self.adj1)
        self.adj2 = nn.Parameter(torch.empty(62, 62))
        nn.init.xavier_normal_(self.adj2)
        self.gatnet1 = GAT(71, 81, 64, 0.15, 0.1, 5)

    def forward(self, x1):
        # self.cuda()
        x1 = self.BN1(x1.transpose(1, 2)).transpose(1, 2)
        x2 = self.sperman(x1).to(device)
        result1 = self.cbam_pro(x1)     # A-CFFEB Model
        result2 = self.sperchannel(x2)  # BCFE Model
        result3 = self.deprocess(x1)    # CFFEB Model
        resultx = torch.cat((result1,  result2), dim=2)
        adj1 = normalize_A(self.adj1)
        adj2 = normalize_A(self.adj2)
        # DGAT-BLS Model
        resulty = self.gatnet1(resultx, adj1, adj2)
        resulty1 = torch.flatten(resulty, 1, 2)
        resulty2 = self.bls1(resulty1)
        result41 = torch.flatten(result3, 1, 2)
        result = torch.cat((resulty2, result41), dim=1)
        result = self.dropout(F.selu(self.fc33(result)))
        result = self.dropout(F.selu(self.fc34(result)))
        result = self.dropout(self.fc35(result))
        return result

