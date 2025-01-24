import sys
import os
root_dir = os.path.abspath(os.getcwd())  # xxx/DG_WSDH
sys.path.append(root_dir)


import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import subprocess
import torch.backends.cudnn as cudnn
import torchmetrics
import time
import argparse
from torch.utils.data import DataLoader
from datasets.datasets import MyDataset, Data_embedding
from model.DG_WSDH import DG_WSDH
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR
from tqdm import tqdm
from collections import defaultdict
from utils.func import get_bag_weight, mask_cross_entropy, weight_cross_entropy, ramp_up, label_patch_create

cudnn.benchmark = True  ##
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--num_workers', type=int, default=24, help='num_workers')
parser.add_argument('--learning_rate1', type=float, default=0.0001, help='Learning rate for branch I')
parser.add_argument('--learning_rate2', type=float, default=0.0001, help='Learning rate for branch II')
parser.add_argument('--bits', type=int, default=64, help='Length of the hash code')
parser.add_argument('--Lambda', type=int, default=5)
parser.add_argument('--num_epoch', type=int, default=25, help='epochs')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0")
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--split_dir', type=str, default='csv/example/split/811', help='Division of the data set')
parser.add_argument('--split_id', type=int, default=0, help='5-fold cross-validation, candidate document name 0-4')
parser.add_argument('--output_dir', type=str, default='example')


par = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = par.CUDA_VISIBLE_DEVICES

output_dir = par.output_dir if par.output_dir else time.strftime('%Y-%m-%d%H', time.localtime(time.time()))
output_dir = os.path.join(root_dir, 'run', output_dir)  #Setting the output file path
os.makedirs(output_dir, exist_ok=True)
subprocess.call(f'cp main.yaml {output_dir}', shell=True)

cp_path = os.path.join(output_dir, 'checkpoint.pth')
if not par.resume and os.path.exists(cp_path):
    os.remove(cp_path)

# Transforms
train_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((1024, 1024)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])
val_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((1024, 1024)),
    T.ToTensor(),
])

# parameter setting
wsi_score = {}

bag_score = {}

up_weight = defaultdict(lambda: 1.0)

wsi_alpha = {}
best_wsi_alpha = {}

bag_alpha = {}
bag_beta = {}
bag_label = {}

best_bag_alpha = {}
best_bag_bet = {}
best_bag_label = {}
best_acc_up = 0
best_acc_down = 0

final_recall = 0
final_precision = 0
final_sp = 0
final_f1 = 0
final_auc = 0

def Branch_I_train(train_loader, epoch, model, optimizer, scheduler = None):
    lossfunc = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight, reduction='none')
    lossfunc.cuda()

    # parameter setting
    num_wsi_train = train_loader.dataset.num_wsi
    temp_embedding = [[] for i in range(num_wsi_train)]
    temp_label = [-1 for i in range(num_wsi_train)]
    temp_bag_id = [[] for i in range(num_wsi_train)]

    total_loss = 0
    t0, t1, t2 = 0, 0, 0
    ramp_up1 = 2 * ramp_up(epoch, par.num_epoch)
    ramp_up2 = par.Lambda ** 2 * ramp_up(epoch, par.num_epoch)

    cnt_one = 0
    model.train()

    # Branch I Training
    for i, (img, wsi_id, bag_id, label, bag_path, bag_weight, index, _, names) in enumerate(tqdm(train_loader, desc=f'train_{epoch}')):
        img = img.cuda()
        label = label.cuda()
        cnt_one += (bag_weight == 1).sum().item()

        bag_weight = bag_weight.cuda()                  # bag_weight
        y, embedding, alpha, y_alpha, beta, y_beta, hashcode = model(img, tag=0, epoch=epoch)  # y: n*2 embedding : n * 512 _ : n*64
        # bag_loss
        loss_bag_branch_I = mask_cross_entropy(y, label, bag_weight, loss=lossfunc)

        w1 = bag_weight.unsqueeze(dim=1) * alpha        # patch weight
        s_ij = label_patch_create(label, names)
        # patch ranking loss
        loss_patch = 1 / 32 * (torch.norm((8 / par.bits * y_alpha - s_ij) * w1.view(-1, 1), p=2, dim=1, keepdim=True)).squeeze(1).mean()

        w2 = (bag_weight.unsqueeze(dim=1) * alpha).view(-1).unsqueeze(dim=1)        # cell weight
        w2 = (w2 * beta).view(-1)
        # cell_loss
        loss_cell = weight_cross_entropy(y_beta.view(-1, 2), label.repeat(64 * 16, 1).T.contiguous().view(-1), w2, loss=lossfunc)
        # all_loss
        loss = loss_bag_branch_I + ramp_up1 * loss_patch + ramp_up2 * loss_cell     # Branch_I all loss
        t0 += loss_bag_branch_I.item()
        t1 += ramp_up1 * loss_patch.item()
        t2 += ramp_up2 * loss_cell.item()

        total_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


        if scheduler is not None:
            scheduler.step()

        for ind in range(len(index)):   # For branch2 training, store branch1's embeddings
            temp_embedding[wsi_id[ind]].append(embedding[ind].tolist())
            temp_label[wsi_id[ind]] = label[ind].item()
            temp_bag_id[wsi_id[ind]].append(bag_id[ind].item())

    print(f"loss0: {t0:.4f}  loss1: {t1:.4f}  loss3: {t2:.4f}")

    return total_loss, temp_embedding, temp_label, temp_bag_id, cnt_one

# Branch_I_evaluate
def Branch_I_evaluate(test_loader, model, epoch):
    num_wsi_test = test_loader.dataset.num_wsi
    embedding_test = [[] for i in range(num_wsi_test)]
    label_test = [-1 for i in range(num_wsi_test)]
    bag_id_test = [[] for i in range(num_wsi_test)]

    correct = 0
    total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (img, wsi_id, bag_id, label, bag_path, bag_weight, index, _, names) in enumerate(test_loader):
            img = img.cuda()
            label = label.cuda()
            y, embedding, alpha, _, beta, __, hashcode = model(img, tag=0,epoch=epoch)  # y: n*2 embedding : n * 512 alpha : n * 64
            loss = F.cross_entropy(y, label, reduction='mean')
            total_loss += loss.item()
            y = y.argmax(dim=1)
            total += y.size(0)
            correct += (y == label).sum().item()

            for ind in range(len(wsi_id)):
                embedding_test[wsi_id[ind]].append(embedding[ind].tolist())
                label_test[wsi_id[ind]] = label[ind].item()
                bag_id_test[wsi_id[ind]].append(bag_id[ind].item())

    acc = correct / total

    return acc, total_loss, embedding_test, label_test, bag_id_test


# Branch II training
def Branch_II_train(train_loader, epoch, net, optimizer, scheduler=None):
    total_loss, t0, t1 = 0, 0, 0
    lossfunc = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight, reduction='none')
    lossfunc.cuda()

    ramp_up3 = par.Lambda * ramp_up(epoch, par.num_epoch)
    up_weight.clear()
    net.train()

    for i, (embedding, label, bag_id, wsi_name, index) in enumerate(train_loader):
        embedding = embedding.cuda()
        label = label.cuda()

        y, alpha, y_alpha = net(embedding, tag=1, epoch=epoch)

        loss_wsi = F.cross_entropy(y, label) if not lossfunc else lossfunc(y, label)
        loss_branch_II_bag = weight_cross_entropy(y_alpha, label.repeat(alpha.shape[0]), alpha, loss=lossfunc)

        loss = loss_wsi + ramp_up3 * loss_branch_II_bag
        t0 += loss_wsi.item()
        t1 += ramp_up3 * loss_branch_II_bag.item()

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        temp = get_bag_weight(wsi_name[0], bag_id.squeeze(dim=0).tolist(), alpha.tolist())
        up_weight.update(temp)

    print(f"_loss0: {t0:.4f}  _loss1: {t1:.4f}")

    scheduler.step()

    return total_loss


# Branch_II_evaluate
def Branch_II_evaluate(test_loader2, net, epoch):
    net.eval()
    correct = 0
    total = 0
    total_loss = 0

    recall_ = torchmetrics.Recall(average='none', num_classes=2)
    precision_ = torchmetrics.Precision(average='none', num_classes=2)
    auc_ = torchmetrics.AUROC(pos_label=1)
    recall_ = recall_.cuda()
    precision_ = precision_.cuda()
    auc_ = auc_.cuda()

    with torch.no_grad():
        for i, (embedding, label, bag_id, wsi_name, index) in enumerate(test_loader2):
            embedding = embedding.cuda()
            label = label.cuda()

            y, alpha, _ = net(embedding, tag=1,epoch=epoch)

            loss = F.cross_entropy(y, label)
            total_loss += loss.item()
            p = y.argmax()
            total += label.size(0)  #
            correct += (p == label).sum().item()

            auc_.update(y.softmax(dim=1)[:, 1], label)
            recall_.update(p.unsqueeze(dim=0), label)
            precision_.update(p.unsqueeze(dim=0), label)

    return correct / total, total_loss, precision_.compute()[1].item(), recall_.compute()[1].item(), \
           auc_.compute().item(), recall_.compute()[0].item()


def Branch_I_dataloader(up_weight=None):
    # Path to get training or val dataset
    pt = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_train.csv')
    pe = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_val.csv')

    train_data = MyDataset(path=pt, transform=train_transforms, up_weight=up_weight)
    val_data = MyDataset(path=pe, transform=val_transforms, up_weight=up_weight)

    # Loader
    train_loader = DataLoader(dataset=train_data, batch_size=par.batch_size, num_workers=par.num_workers, drop_last=True, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=par.batch_size, num_workers=par.num_workers, drop_last=True, shuffle=False, pin_memory=True)

    return train_loader, val_loader


def Branch_II_dataloader(embedding, label, bag_id, wsi_name):
    # Load the embeddings stored from branch_I
    data2 = Data_embedding(data=(embedding, label, bag_id, wsi_name))
    dataloader2 = DataLoader(dataset=data2, batch_size=1, shuffle=True)
    return dataloader2


def Branch_II_task(train_loader2, val_loader2, net, optimizer2, scheduler2, epoch_up):
    id_up = f'{epoch_up:0{2}}'
    loss = Branch_II_train(train_loader2, epoch_up, net, optimizer2, scheduler2)
    acc, loss_val, precision, recall, auc, sp = Branch_II_evaluate(val_loader2, net, epoch=epoch_up)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    print(f'_epoch_{id_up}:  train loss: {loss:.4f}      val loss: {loss_val:.4f}      val acc: {acc:.6f} '
          f'auc: {auc:.6f}   f1: {f1:.6f}    se: {recall:.6f}       sp: {sp:.6f}')

    return acc, recall, precision, sp, f1, auc

def get_path(save_dir):
    a_path = os.path.join(save_dir, 'alpha.pt')
    cp_path = os.path.join(save_dir, 'checkpoint.pth')
    config_path = os.path.join(save_dir, 'main.yaml')
    res_path = os.path.join(save_dir, 'result.pth')
    pred_path = os.path.join(save_dir, 'predict.csv')
    return a_path, cp_path, config_path, res_path, pred_path



def train_main():
    global up_weight
    # Initialise branch_I's loader
    Branch_I_train_loader, Branch_I_val_loader = Branch_I_dataloader(up_weight)
    net = DG_WSDH()
    net.cuda()
    net.upstream = torch.nn.DataParallel(net.upstream, device_ids=range(torch.cuda.device_count()))
    # Setting up the Optimiser
    optimizer1 = torch.optim.Adam(net.upstream.parameters(), lr=par.learning_rate1)
    scheduler1 = OneCycleLR(optimizer1, max_lr=par.learning_rate1, steps_per_epoch=len(Branch_I_train_loader), epochs=par.num_epoch)
    optimizer2 = torch.optim.Adam(net.downstream.parameters(), lr=par.learning_rate2)
    scheduler2 = MultiStepLR(optimizer2, milestones=[5], gamma=.5)

    global best_acc_up, best_acc_down, bag_alpha, final_auc, final_recall, final_f1, final_precision
    best_acc_down_epoch = -1

    start_epoch = 0
    # Determine whether to restart training based on the parameters
    if par.resume and os.path.exists(cp_path):
        cp = torch.load(cp_path)
        net.load_state_dict(cp['model_state_dict'])
        optimizer1.load_state_dict(cp['optimizer1_state_dict'])
        scheduler1.load_state_dict(cp['scheduler1_state_dict'])
        optimizer2.load_state_dict(cp['optimizer2_state_dict'])
        scheduler2.load_state_dict(cp['scheduler2_state_dict'])
        start_epoch = cp['epoch'] + 1
        up_weight = defaultdict(lambda: 1.0, cp['up_weight'])
        Branch_I_train_loader, Branch_I_val_loader = Branch_I_dataloader(up_weight)

    for epoch in range(start_epoch, par.num_epoch):
        print(epoch, par.num_epoch)
        # Branch_I's train and val
        loss, temp_embedding, temp_label, temp_bag_id, _ = Branch_I_train(Branch_I_train_loader, epoch, net, optimizer1, scheduler1)
        acc, loss_val, embedding_val, label_val, bag_id_val = Branch_I_evaluate(Branch_I_val_loader, net, epoch=epoch)

        if acc > best_acc_up:
            best_acc_up = acc

        print(f'epoch: {epoch} train loss: {loss:.2f}     val loss: {loss_val:.2f}     val acc: {acc:.6f}')

        # Branch_II's train and val
        Branch_II_train_loader = Branch_II_dataloader(temp_embedding, temp_label, temp_bag_id, Branch_I_train_loader.dataset.wsi_name)
        Branch_II_val_loader = Branch_II_dataloader(embedding_val, label_val, bag_id_val, Branch_I_val_loader.dataset.wsi_name)

        _acc, recall, precision, sp, f1, auc = Branch_II_task(Branch_II_train_loader, Branch_II_val_loader, net, optimizer2, scheduler2, epoch)

        if _acc >= best_acc_down:
            best_acc_down = _acc
            best_acc_down_epoch = epoch
            final_recall = recall
            final_precision = precision
            final_sp = sp
            final_f1 = f1
            final_auc = auc
            best_pth = net.state_dict()

        torch.save({'id': id, 'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer1_state_dict': optimizer1.state_dict(),
                    'scheduler1_state_dict': scheduler1.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'scheduler2_state_dict': scheduler2.state_dict(),
                    'up_weight': dict(up_weight),
                    'best_pth': best_pth}, cp_path)

        Branch_I_train_loader, Branch_I_val_loader = Branch_I_dataloader(up_weight)

    print(f'best acc_up: {best_acc_up:.4f}     best acc_down: {best_acc_down:.4f}   final_f1: {final_f1:.4f}   '
          f'final_auc: {final_auc:.4f}    best_acc_epoch:{best_acc_down_epoch:.4f}')

    print(f'training completed.\n checkpoint is saved in output_dir: {output_dir}')


def test(model,res_path):
    #  Path to get test dataset
    p_test = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_test.csv')
    test_data = MyDataset(p_test, transform=val_transforms)
    test_loader = DataLoader(dataset=test_data, batch_size=par.batch_size, num_workers=par.num_workers, drop_last=True, shuffle=False, pin_memory=True)

    up_acc, loss_test, embedding_test, label_test, bag_id_test = test_branch_I_evaluate(test_loader, model, None)
    test_loader2 = Branch_II_dataloader(embedding_test, label_test, bag_id_test, test_loader.dataset.wsi_name)
    acc, loss_test, precision, recall, auc, sp = test_branch_II_evaluate2(test_loader2, model, None)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    torch.save(dict(acc=acc, auc=auc, f1=f1, se=recall, sp=sp), res_path)
    print(f"test_acc: {acc:.4f}  test_auc: {auc:.4f}    test_f1: {f1:.4f}   test_precision: {precision:.4f}  test_se: {recall:.4f}  test_sp:{sp:.4f}")

def test_main():
    # get path
    alpha_path, cp_path, config_path, res_path, pred_path = get_path(output_dir)
    net = DG_WSDH()
    net.cuda()
    net.upstream = torch.nn.DataParallel(net.upstream, device_ids=range(torch.cuda.device_count()))
    # load best check_point
    cp = torch.load(cp_path)
    net.load_state_dict(cp['best_pth'])

    test(net, res_path)

    torch.save(dict(best_bag_alpha=bag_alpha, best_bag_score=bag_score, best_bag_beta=bag_beta,
                    best_wsi_alpha=wsi_alpha, best_wsi_scrore=wsi_score, best_bag_label=bag_label), alpha_path)

def test_branch_II_evaluate2(test_loader2, net, epoch):
    net.eval()

    correct = 0
    total = 0
    total_loss = 0

    recall_ = torchmetrics.Recall(average='none', num_classes=2)
    precision_ = torchmetrics.Precision(average='none', num_classes=2)
    auc_ = torchmetrics.AUROC(pos_label=1)
    recall_ = recall_.cuda()
    precision_ = precision_.cuda()
    auc_ = auc_.cuda()

    with torch.no_grad():
        for i, (embedding, label, bag_id, wsi_name, index) in enumerate(test_loader2):
            embedding = embedding.cuda()
            label = label.cuda()
            bag_id = bag_id.cuda()
            y, alpha, _ = net(embedding, tag=1, epoch=20)

            loss = F.cross_entropy(y, label)
            total_loss += loss.item()
            p = y.argmax()
            total += label.size(0)  #
            correct += (p == label).sum().item()

            auc_.update(y.softmax(dim=1)[:, 1], label)
            recall_.update(p.unsqueeze(dim=0), label)
            precision_.update(p.unsqueeze(dim=0), label)
    return correct / total, total_loss, precision_.compute()[1].item(), recall_.compute()[
        1].item(), auc_.compute().item(), recall_.compute()[0].item()

def test_branch_I_evaluate(test_loader, model, epoch):
    num_wsi_test = test_loader.dataset.num_wsi
    embedding_test = [[] for i in range(num_wsi_test)]
    label_test = [-1 for i in range(num_wsi_test)]
    bag_id_test = [[] for i in range(num_wsi_test)]

    correct = 0
    total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (img, wsi_id, bag_id, label, bag_path, bag_weight, index, _,names) in enumerate(test_loader):
            img = img.cuda()
            label = label.cuda()
            y, embedding, alpha, _1, beta, _3, __ = model(img, tag=0, epoch=20)  # y: n*2 embedding : n * 512 alpha : n * 64
            loss = F.cross_entropy(y, label, reduction='mean')
            total_loss += loss.item()
            y = y.argmax(dim=1)
            total += y.size(0)
            correct += (y == label).sum().item()

            for ind in range(len(wsi_id)):
                embedding_test[wsi_id[ind]].append(embedding[ind].tolist())
                label_test[wsi_id[ind]] = label[ind].item()
                bag_id_test[wsi_id[ind]].append(bag_id[ind].item())


    acc = correct / total
    return acc, total_loss, embedding_test, label_test, bag_id_test

if __name__ == '__main__':
    train_main()
    print('testing')
    test_main()
