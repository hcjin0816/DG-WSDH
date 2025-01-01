import torch.nn.functional as F
import torch
import wandb
from math import exp


def ramp_up(epoch, num_epoch):
    if num_epoch == 1:
        return 1
    T = epoch / (num_epoch - 1)
    x = exp(-(1 - T) ** 2)
    return x

def mask_cross_entropy(y, label, mask, loss=None):
    if loss is None:
        l = F.cross_entropy(y, label, reduction='none') * mask
    else:
        l = loss(y, label) * mask
    return l.mean()


def weight_cross_entropy(y, label, w, reduction='none', loss=None):
    if loss is None:
        l = F.cross_entropy(y, label, reduction='none') * w
    else:
        l = loss(y, label) * w
    return l.mean() if reduction != 'sum' else l.sum()


def pred_lr(dataloader, net):
    wandb.init(project="GDC", entity="xhc")
    lr = 1e-6
    temp = 0
    for i, (img, wsi_id, label, index) in enumerate(dataloader):
        if lr > 1:
            break
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        img = img.cuda()
        label = label.cuda()

        y, _ = net(img)
        loss = F.cross_entropy(y, label.repeat((64, 1)).T.flatten(), reduction='mean')
        temp += loss.item()
        if i % 50 == 0:
            wandb.log({'loss': temp, 'learing rate': lr})
            temp = 0
            lr *= 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_bag_weight(wsi_name: str, bag_id: list, alpha: list):
    res = {}
    n = len(bag_id)
    for i, w in zip(bag_id, alpha):
        p2 = wsi_name.split('/')[-1] + '_' + str(i)
        res[p2] = w * n

    return res


def save_attention(bag_score, bag_alpha, bag_beta, bag_path, alpha, beta, bag_label, label, num_ceil=16):
    beta = beta.view(-1, 64, num_ceil)

    for alpha_i, beta_i, path_i, label_i in zip(alpha, beta, bag_path, label):
        ind = alpha_i.argsort(descending=True)
        bag_alpha[path_i] = ind.tolist()
        bag_score[path_i] = alpha_i.tolist()
        bag_label[path_i] = label_i

        ind_cid = []
        for pid in ind:
            # cid = beta_i[pid].argsort(descending=True)
            ind_cid.append(beta_i[pid].tolist())
        bag_beta[path_i] = ind_cid

def average_precision_patch(query, ranking):
    query = query
    scores = []
    j = 0
    index = 0
    for i in range(len(ranking)):
        index += 1
        if query == ranking[i]:
            j += 1
            score = j / index
            scores.append(score)

    if j != 0:
        return sum(scores) / len(scores)
    else:
        return 0

sij_same_1 = torch.full([64, 64], 6)
sij_same_2 = torch.full([64, 64], 8)
sij_no = torch.full([64, 64], -1)

# Sij is set according to the label of the WSI to which the patch belongs.
def label_patch_create(label, names):
    if names[0] == names[1]: # Same WSI
        sij_same = sij_same_2
    else:
        sij_same = sij_same_1
    sij = []
    for i in range(len(label)):
        temp = label[i]
        y = sij_same if temp == label[0] else sij_no
        for j in range(1, len(label)):
            if temp == label[j]:
                y = torch.cat((y, sij_same), dim=1)
            else:
                y = torch.cat((y, sij_no), dim=1)
        sij.append(y)
    s = torch.cat(sij).cuda()
    return s