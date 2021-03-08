import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torchvision import transforms
from data.multi_minist import MultiMNIST
from net.lenet import MultiLeNetR, MultiLeNetO
from pcgrad import PCGrad

# ------------------ CHANGE THE CONFIGURATION -------------
PATH = './dataset'
LR = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 100
TASKS = ['R', 'L']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------------------------------------------


def accuracy(logits, gt):
    # pdb.set_trace()
    return ((logits.argmax(dim=-1) == gt).float()).mean()

def get_model(params=None):
    model = {
        'rep': MultiLeNetR().to(DEVICE),
        'L': MultiLeNetO().to(DEVICE),
        'R': MultiLeNetO().to(DEVICE)
    }
    # if 'L' in params['tasks']:
    #     model['L'] = MultiLeNetO()
    #     model['L'].cuda()
    # if 'R' in params['tasks']:
    #     model['R'] = MultiLeNetO()
    #     model['R'].cuda()
    return model


to_dev = lambda inp, dev: [x.to(dev) for x in inp]

global_transformer = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))])

train_dst = MultiMNIST(PATH,
                       train=True,
                       download=True,
                       transform=global_transformer,
                       multi=True)
train_loader = torch.utils.data.DataLoader(train_dst,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=4)

val_dst = MultiMNIST(PATH,
                     train=False,
                     download=True,
                     transform=global_transformer,
                     multi=True)
val_loader = torch.utils.data.DataLoader(val_dst,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=4)
nets = {
    'rep': MultiLeNetR().to(DEVICE),
    'L': MultiLeNetO().to(DEVICE),
    'R': MultiLeNetO().to(DEVICE)
}
param = [p for v in nets.values() for p in list(v.parameters())]
optimizer = torch.optim.Adam(param, lr=LR)
# optimizer = torch.optim.Adam({k: v.parameters()
#                               for k, v in nets.items()},
#                              lr=LR)
optimizer = PCGrad(optimizer)

for ep in range(NUM_EPOCHS):
    for net in nets.values():
        net.train()
    for batch in train_loader:
        mask = None
        optimizer.zero_grad()
        img, label_l, label_r = to_dev(batch, DEVICE)
        rep, mask = nets['rep'](img, mask)
        out_l, mask_l = nets['L'](rep, None)
        out_r, mask_r = nets['R'](rep, None)

        # pdb.set_trace()
        losses = [F.nll_loss(out_l, label_l), F.nll_loss(out_r, label_r)]
        optimizer.pc_backward(losses)
        optimizer.step()

    for net in nets.values():
        net.eval()
    for batch in val_loader:
        img, label_l, label_r = to_dev(batch, DEVICE)
        mask = None
        rep, mask = nets['rep'](img, mask)
        out_l, mask_l = nets['L'](rep, None)
        out_r, mask_r = nets['R'](rep, None)

        losses = [
            F.nll_loss(out_l, label_l).item(),
            F.nll_loss(out_r, label_r).item()
        ]
        acc = [
            accuracy(out_l, label_l).item(),
            accuracy(out_r, label_r).item()
        ]
    print(losses)
    print(acc)
