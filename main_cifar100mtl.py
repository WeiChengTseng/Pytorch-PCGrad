import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import os
import tqdm

from pcgrad import PCGrad
from data.cifar100mtl import CIFAR100MTL
from net.cifar100 import RoutedAllFC
from net.routing_net.rl import WPL
from utils import create_logger


def compute_batch(model, batch):
    samples, labels, tasks = batch
    out, meta = model(samples, tasks=tasks)
    correct_predictions = (out.max(
        dim=1)[1].squeeze() == labels.squeeze()).cpu().numpy()
    accuracy = correct_predictions.sum()
    oh_labels = one_hot(labels, out.size()[-1])
    module_loss, decision_loss = model.loss(out, meta, oh_labels)
    return module_loss, decision_loss, accuracy


def one_hot(indices, width):
    indices = indices.squeeze().unsqueeze(1)
    oh = torch.zeros(indices.size()[0], width).to(indices.device)
    oh.scatter_(1, indices, 1)
    return oh


# ------------------ CHANGE THE CONFIGURATION -------------
PATH = './dataset/cifar-100-python'
LR = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 50
TASKS = []
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------------------------------------------


logger = create_logger('Main')

dataset = CIFAR100MTL(
    10,
    data_files=[os.path.join(PATH, 'train'),
                os.path.join(PATH, 'test')],
    cuda=torch.cuda.is_available())
model = RoutedAllFC(WPL, 3, 640, 5, dataset.num_tasks,
                    dataset.num_tasks).to(DEVICE)
# model = RoutedAllFC(WPL, 3, 128, 5, dataset.num_tasks,
#                     dataset.num_tasks).to(DEVICE)

learning_rates = {0: 3e-3, 5: 1e-3, 10: 3e-4}
routing_module_learning_rate_ratio = 0.3

# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# optimizer = PCGrad(optimizer)

logger.info('Loaded dataset and constructed model. Starting Training ...')
for epoch in range(NUM_EPOCHS):
    optimizers, parameters = [], []
    if epoch in learning_rates:
        try:
            optimizers.append(
                torch.optim.SGD(model.routing_parameters(),
                                lr=routing_module_learning_rate_ratio *
                                learning_rates[epoch]))
            optimizers.append(
                torch.optim.SGD(model.module_parameters(),
                                lr=learning_rates[epoch]))
            parameters = model.module_parameters() + model.module_parameters()
        except AttributeError:
            optimizers.append(
                torch.optim.SGD(model.parameters(), lr=learning_rates[epoch]))
            parameters = model.parameters()
    train_log, test_log = np.zeros((3, )), np.zeros((3, ))
    train_samples_seen, test_samples_seen = 0, 0
    dataset.enter_train_mode()
    model.train()
    pbar = tqdm.tqdm(unit=' samples')
    while True:
        try:
            batch = dataset.get_batch()
        except StopIteration:
            break
        train_samples_seen += len(batch[0])
        pbar.update(len(batch[0]))
        module_loss, decision_loss, accuracy = compute_batch(model, batch)
        (module_loss + decision_loss).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 40., norm_type=2)
        for opt in optimizers:
            opt.step()
        model.zero_grad()
        train_log += np.array(
            [module_loss.tolist(),
             decision_loss.tolist(), accuracy])
    pbar.close()
    dataset.enter_test_mode()
    model.eval()
    model.start_logging_selections()
    while True:
        try:
            batch = dataset.get_batch()
        except StopIteration:
            break
        test_samples_seen += len(batch[0])
        module_loss, decision_loss, accuracy = compute_batch(model, batch)
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
    model.stop_logging_selections_and_report()
