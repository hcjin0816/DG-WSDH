import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class branch_1_module(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=2, keep_patch_threshold=True,
                 top_patch_num=None):
        super(branch_1_module, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.num_classes = num_classes
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.attention_cell = attention_cell()
        self.attention_patch = attention_patch(keep_patch_threshold=keep_patch_threshold, top_patch_num=top_patch_num)
        self.GCN = pre_GCN(512) # Setting the GCN Parameters

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, epoch, patch_alpha):
        n = x.shape[0]
        out = rearrange(x, 'N C (ph h) (pw w) -> (N ph pw) C h w', ph=8, pw=8)  # (N*64)*3*128*128 (128, 3. 128, 128)

        out = F.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # (N*64)*512*4*4
        # (128. 512. 4. 4)
        out, beta, y_beta = self.attention_cell(out, self.linear.weight, self.linear.bias)
        # # (128, 512) beta(128, 16) y_beta(128, 16,2)
        out = out.reshape(n, -1, out.shape[-1])
        # (2, 64, 512)
        out = self.GCN(out)
        out, embedding, alpha, out_alpha,hash_codes = self.attention_patch(out, self.linear.weight, self.linear.bias, patch_alpha)
        # out's shape: n * num_classes       embedding's shape: n * 512     alpha' shape: 64
        return out, embedding, alpha, out_alpha, beta, y_beta,hash_codes
        # (2,2)  bag_embedding(2, 512) alpha (2,64)  out_a(2,64,2)

class attention_cell(nn.Module):
    def __init__(self, ):
        super(attention_cell, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x, w, b):
        # w, b, 最后分类linear
        # w_1,b_1, 降维linear
        #  x: (n*64) x 512 x 16 x 16
        x = x.flatten(start_dim=2)
        x = x.permute(0, 2, 1)  # (n * 64) * 256 * 512
        y_beta = F.linear(x, w, b)
        gamma = x.shape[1]
        alpha = torch.sqrt((x ** 2).sum(dim=2))
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        alpha = F.relu(alpha - .3 / float(gamma))  # Remove trival
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        out = alpha.unsqueeze(dim=2) * x
        out = out.sum(dim=1)

        return out, alpha, y_beta

class attention_patch(nn.Module):
    def __init__(self, keep_patch_threshold=True, top_patch_num=None):
        super(attention_patch, self).__init__()
        self.keep_patch_threshold = keep_patch_threshold
        self.top_patch_num = top_patch_num
        self.act = nn.ReLU()
        self.hash = nn.Linear(512, 64)
    def forward(self, x, w, b, patch_alpha):
        if not self.keep_patch_threshold:
            x_ = torch.empty((x.shape[0], self.top_patch_num, x.shape[2]), device=x.device)
            for i in range(x.shape[0]):
                x_[i] = x[i, patch_alpha[i][:self.top_patch_num]]
            x = x_
        gamma = x.shape[1]

        # hash
        hash_codes = torch.sign(self.hash(x))
        out_alpha = torch.mm(hash_codes.reshape(-1, 64), hash_codes.reshape(-1, 64).t())
        # (128, 128)

        out = torch.sqrt((x ** 2).sum(dim=2))
        alpha = out / out.sum(dim=1, keepdim=True).expand_as(out)
        if self.keep_patch_threshold:
            alpha = F.relu(alpha - .2 / float(gamma))  # Remove trival
        alpha = alpha / alpha.sum(dim=1, keepdim=True).expand_as(alpha)
        # embedding
        out = alpha.unsqueeze(dim=2).expand_as(x) * x
        out = out.sum(dim=1)
        embedding = out.detach()
        out = F.linear(out, w, b)

        alpha = alpha.detach()
        # out's shape: n * num_classes   embedding's shape: n * 512     out_alpha's shape: n * 64 * num_classes
        return out, embedding, alpha, out_alpha,hash_codes


class pre_GCN(nn.Module):
    def __init__(self, dim):
        super(pre_GCN, self).__init__()
        self.temp_matrix = torch.full([dim, dim], 1.0, requires_grad=True).cuda()
        self.kminc = 1
        self.V0 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.V1 = nn.Linear(dim, 128)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=5e-2)
        self.GCN = GCN_patch(dim, dim, dim)

    def forward(self,x):
        n = x.shape[0]
        for i in range(n):
            Z = torch.mm(x[i].clone(), self.V1(self.act(self.V0(self.temp_matrix))))
            Q = torch.mm(self.LeakyReLU(Z), torch.t(self.LeakyReLU(Z)))
            Q = Q - self.kminc * torch.mean(Q)
            Q = F.relu(Q)
            Q = F.normalize(Q, dim=0)
            Q = self.process_adj(Q + (1e-7) * torch.eye(Q.shape[0]).cuda())
            x[i] = self.GCN(x[i], Q)
        return x

    def process_adj(self, adj):
        adj = (adj + torch.t(adj)) / 2
        rowsum = adj.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        adj = torch.matmul(adj, r_mat_inv_sqrt)
        adj = torch.transpose(adj, 0, 1)
        adj = torch.matmul(adj, r_mat_inv_sqrt)
        return adj

class attention_bag(nn.Module):
    def __init__(self, keep_threshold=False):
        super(attention_bag, self).__init__()
        self.hash = nn.Linear(512, 64)
        self.classifier = nn.Linear(64, 2)
        self.down_fc = nn.Linear(512, 64)
        self.keep_threshold = keep_threshold

    def forward(self, x, w, b):
        # Process bag in one slide at a time
        # x = torch.squeeze(x, dim=0)
        gamma = x.shape[1]
        out = self.down_fc(x)
        # linear (512->64)
        out_alpha = F.linear(out, w, b)
        # linear (64->2)

        out = torch.sqrt((out_alpha ** 2).sum(dim=2))
        alpha = out / out.sum(dim=1, keepdim=True).expand_as(out)
        if self.keep_threshold:
            alpha = F.relu(alpha - .1 / float(gamma))
        alpha = alpha / alpha.sum(dim=1, keepdim=True).expand_as(alpha)

        # WSI_p
        out = alpha.unsqueeze(dim=2).expand_as(x) * x
        out = out.sum(dim=1)
        out = torch.sign(self.hash(out))
        out = F.linear(out, w, b)

        alpha = alpha.squeeze(dim=0).detach()
        # out's shape: 1* num_classes   alpha's shape: gamma    out_alpha: 1 * gamma * num_classes
        return out, alpha, out_alpha.squeeze(dim=0)


class branch_2_module(nn.Module):
    def __init__(self, num_classes=2, num_hidden_unit=32, keep_threshold=True):
        super(branch_2_module, self).__init__()
        self.linear0 = nn.Linear(512, 128)
        self.linear1 = nn.Linear(64, num_classes)
        self.attention = attention_bag(keep_threshold=keep_threshold)
    def forward(self, x, epoch):
        return self.attention(x, self.linear1.weight, self.linear1.bias)


class DG_WSDH(nn.Module):
    def __init__(self, num_classes=2, num_hidden_unit=32, keep_bag_threshold=True,
                 keep_patch_threshold=True, top_patch_num=None):
        super(DG_WSDH, self).__init__()
        self.upstream = branch_1_module(num_classes=num_classes, keep_patch_threshold=keep_patch_threshold,
                                        top_patch_num=top_patch_num)  # todo
        self.downstream = branch_2_module(num_hidden_unit=num_hidden_unit, keep_threshold=keep_bag_threshold)

    def forward(self, x, tag, epoch, patch_alpha=-1):
        if tag == 0:
            return self.upstream(x, epoch, patch_alpha=patch_alpha)
        else:
            return self.downstream(x, epoch)


class conv(nn.Module):
    def __init__(self, num_ceil=64):
        super(conv, self).__init__()
        ceil_edge = int(num_ceil ** .5)
        assert ceil_edge ** 2 == num_ceil and ceil_edge % 2 == 0 and ceil_edge <= 16

        cnt = int((16 - ceil_edge) / 2)
        li = [conv_unit() for i in range(cnt)]
        self.conv = nn.Sequential(*li)

    def forward(self, x):
        return self.conv(x)


class conv_unit(nn.Module):
    def __init__(self):
        super(conv_unit, self).__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class GCN_patch(nn.Module):
    def __init__(self, channels=512, hid_features=512, out_features=512):
        super(GCN_patch, self).__init__()
        self.GCNconv1 = GraphConvolutionLayer(channels, hid_features)
        self.GCNconv2 = GraphConvolutionLayer(hid_features, out_features)

    def forward(self, x, adj):
        # X (1,64,512)
        # 将所输入的64个patch建图
        # sklearn 输入批次特征，n为K, p=2为欧氏距离
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.GCNconv1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.GCNconv2(x, adj)
        return x


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 参数均匀初始化
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_w = torch.mm(x, self.weight)
        output = torch.spmm(adj, x_w)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 自我描述的信息
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
