import sys
import os

root_dir = os.path.abspath(os.getcwd())  # xxx/MRAN
sys.path.append(root_dir)
import argparse
import csv
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import subprocess
import torch.backends.cudnn as cudnn
import torchmetrics
import time
from torch.utils.data import DataLoader
from datasets.datasets import MyDataset, Data_embedding
from Ablation.DG_WSDH.model.ResNet import DG_WSDH
from tqdm import tqdm
from collections import defaultdict

cudnn.benchmark = True  ##
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--num_workers', type=int, default=36, help='num_workers')
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
output_dir = os.path.join(root_dir, 'run', output_dir) #Setting the output file path
os.makedirs(output_dir, exist_ok=True)
subprocess.call(f'cp main.yaml {output_dir}', shell=True)

cp_path = os.path.join(output_dir, 'checkpoint.pth')
if not par.resume and os.path.exists(cp_path):
    os.remove(cp_path)


train_transforms = T.Compose([
    T.ToTensor(),
])
val_transforms = T.Compose([
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

def Branch_II_dataloader(embedding, label, bag_id, wsi_name):
    # Load the embeddings stored from branch_I
    data2 = Data_embedding(data=(embedding, label, bag_id, wsi_name))
    dataloader2 = DataLoader(dataset=data2, batch_size=1, shuffle=True)
    return dataloader2



def test(model,res_path):
    #  Path to get test dataset
    p_test = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_train.csv')
    test_data = MyDataset(p_test, transform=val_transforms)
    test_loader = DataLoader(dataset=test_data, batch_size=par.batch_size, num_workers=par.num_workers, drop_last=True, shuffle=False, pin_memory=True)

    up_acc, loss_test, embedding_test, label_test, bag_id_test =test_branch_I_evaluate(test_loader, model, None)
    test_loader2 = Branch_II_dataloader(embedding_test, label_test, bag_id_test, test_loader.dataset.wsi_name)
    acc, loss_test, precision, recall, auc, sp = test_branch_II_evaluate2(test_loader2, model, None)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    torch.save(dict(acc=acc, auc=auc, f1=f1, se=recall, sp=sp), res_path)
    print(f"test_acc: {acc:.4f}  test_auc: {auc:.4f}    test_f1: {f1:.4f}   test_precision: {precision:.4f}  test_se: {recall:.4f}  test_sp:{sp:.4f}")


def get_path(save_dir):
    a_path = os.path.join(save_dir, 'alpha.pt')
    cp_path = os.path.join(save_dir, 'checkpoint.pth')
    config_path = os.path.join(save_dir, 'main.yaml')
    res_path = os.path.join(save_dir, 'result.pth')
    pred_path = os.path.join(save_dir, 'predict.csv')
    return a_path, cp_path, config_path, res_path, pred_path


def test_branch_I_evaluate(test_loader, model, epoch):
    num_wsi_test = test_loader.dataset.num_wsi
    embedding_test = [[] for i in range(num_wsi_test)]
    label_test = [-1 for i in range(num_wsi_test)]
    bag_id_test = [[] for i in range(num_wsi_test)]

    num_wsi_train = test_loader.dataset.num_wsi
    patch_hash_codes = [[] for i in range(num_wsi_train)]
    wsi_name = test_loader.dataset.wsi_name
    wsi_bag_id = [[] for i in range(num_wsi_train)]
    wsi_bag_patch_label = [[] for i in range(num_wsi_train)]
    patch_hash_codes_online = [[] for i in range(num_wsi_train)]
    wsi_bag_patch_label_online = [[] for i in range(num_wsi_train)]
    correct = 0
    total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (img, wsi_id, bag_id, label, bag_path, bag_weight, index, _, patch_label,namse) in enumerate(
                tqdm(test_loader, desc=f'train_{epoch}')):
            img = img.cuda()
            label = label.cuda()
            # ResNet中需要将tanh换成sign函数，生成二进制哈希码进行存储
            y, embedding, alpha, _1, beta, _3, hashcodes = model(img, tag=0, epoch=epoch)  # y: n*2 embedding : n * 512 alpha : n * 64
            loss = F.cross_entropy(y, label, reduction='mean')
            total_loss += loss.item()
            y = y.argmax(dim=1)
            total += y.size(0)
            correct += (y == label).sum().item()

            hashcodes = hashcodes.view(2, 64, 64)
            for ind in range(len(index)):
                if label[ind].item() == 1:
                    patch_hash_codes_online[wsi_id[ind]].append(hashcodes[ind])
                    wsi_bag_patch_label_online[wsi_id[ind]].append(patch_label[ind].tolist())
            for ind in range(len(wsi_id)):
                embedding_test[wsi_id[ind]].append(embedding[ind].tolist())
                label_test[wsi_id[ind]] = label[ind].item()
                bag_id_test[wsi_id[ind]].append(bag_id[ind].item())

            for ind in range(len(index)):
                patch_hash_codes[wsi_id[ind]].append(hashcodes[ind])
                wsi_bag_patch_label[wsi_id[ind]].append(patch_label[ind].tolist())

    for i in range(len(wsi_bag_id)):
        patch_hash_codes[i] = torch.stack(patch_hash_codes[i]).view(-1, 64)

    # save path
    torch.save({'hashcodes': patch_hash_codes_online, 'patch_label': wsi_bag_patch_label_online}, '.pt')

    acc = correct / total
    return acc, total_loss, embedding_test, label_test, bag_id_test


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

    WSI_hashcodes = []
    wsi_temp_label = []
    with torch.no_grad():
        for i, (embedding, label, bag_id, wsi_name, index) in enumerate(test_loader2):
            embedding = embedding.cuda()
            label = label.cuda()
            y, alpha, _, hash_codes = net(embedding, tag=1, epoch=20)

            loss = F.cross_entropy(y, label)
            total_loss += loss.item()
            p = y.argmax()
            total += label.size(0)  #
            correct += (p == label).sum().item()

            auc_.update(y.softmax(dim=1)[:, 1], label)
            recall_.update(p.unsqueeze(dim=0), label)
            precision_.update(p.unsqueeze(dim=0), label)
            WSI_hashcodes.append(hash_codes)
            wsi_temp_label.append(label.item())

    wsi_hashcodes = {'hashcodes': torch.stack(WSI_hashcodes).view(-1, 64), 'label': torch.tensor(wsi_temp_label)}
    torch.save(wsi_hashcodes, '')   # save pt path

    return correct / total, total_loss, precision_.compute()[1].item(), recall_.compute()[
            1].item(), auc_.compute().item(), recall_.compute()[0].item()


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


if __name__ == '__main__':
    test_main()

