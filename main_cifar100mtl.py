import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import os

from pcgrad import PCGrad
from data.cifar100mtl import CIFAR100MTL
from net.cifat100 import Net

# ------------------ CHANGE THE CONFIGURATION -------------
PATH = './dataset/cifar-100-python'
LR = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 50
TASKS = []
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------------------------------------------

net = Net().to(DEVICE)
dataset = CIFAR100MTL(
    10,
    data_files=[os.path.join(PATH, 'train'),
                os.path.join(PATH, 'test')],
    cuda=torch.cuda.is_available())

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
optimizer = PCGrad(optimizer)

for ep in range(NUM_EPOCHS):
    dataset.enter_train_mode()
    net.train()

    while True:
        try:
            batch = dataset.get_batch()
        except StopIteration:
            break
        pdb.set_trace()

        
        train_samples_seen += len(batch[0])
        pbar.update(len(batch[0]))
        module_loss, decision_loss, accuracy = compute_batch(net, batch)
        (module_loss + decision_loss).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 40., norm_type=2)
        for opt in optimizers:
            opt.step()
        net.zero_grad()
        train_log += np.array(
            [module_loss.tolist(),
             decision_loss.tolist(), accuracy])

    dataset.enter_test_mode()
    net.eval()
    net.start_logging_selections()
    while True:
        try:
            batch = dataset.get_batch()
        except StopIteration:
            break
        test_samples_seen += len(batch[0])
        module_loss, decision_loss, accuracy = compute_batch(net, batch)
        test_log += np.array(
            [module_loss.tolist(),
             decision_loss.tolist(), accuracy])
    print(
        'Epoch {} finished after {} train and {} test samples..\n'
        '    Training averages: Model loss: {}, Routing loss: {}, Accuracy: {}\n'
        '    Testing averages:  Model loss: {}, Routing loss: {}, Accuracy: {}'
        .format(epoch + 1, train_samples_seen, test_samples_seen,
                *(train_log / train_samples_seen).round(3),
                *(test_log / test_samples_seen).round(3)))
    net.stop_logging_selections_and_report()
