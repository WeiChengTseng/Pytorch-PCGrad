import abc
import random
import pickle

import numpy as np

import torch
from torch.autograd import Variable


class Dataset(object, metaclass=abc.ABCMeta):
    """
    Class Datasets defines ...
    """
    def __init__(self, batch_size, data_files=(), cuda=False):
        self._iterator = None
        self._batch_size = batch_size
        self._data_files = data_files
        self._train_set, self._test_set = self._get_datasets()
        self._cuda = cuda

    @abc.abstractmethod
    def _get_datasets(self):
        return [], []

    def _batched_iter(self, dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            samples = (torch.stack(
                [torch.FloatTensor(sample[0]) for sample in batch], 0))
            targets = (torch.stack(
                [torch.LongTensor([sample[1]]) for sample in batch], 0))
            if self._cuda:
                samples, targets = samples.cuda(), targets.cuda()
            tasks = [sample[2] for sample in batch]
            yield samples, targets, tasks

    def get_batch(self):
        return next(self._iterator)

    def enter_train_mode(self):
        random.shuffle(self._train_set)
        self._iterator = self._batched_iter(self._train_set, self._batch_size)

    def enter_test_mode(self):
        self._iterator = self._batched_iter(self._test_set, self._batch_size)


class CIFAR100MTL(Dataset):
    def __init__(self, *args, **kwargs):
        Dataset.__init__(self, *args, **kwargs)
        self.num_tasks = 20
        return

    def _get_datasets(self):
        datasets = []
        for fn in self._data_files:
            # assuming that the datafiles are [train_file_name, test_file_name]
            samples, labels, tasks = [], [], []
            with open(fn, 'rb') as f:
                data_dict = pickle.load(f, encoding='latin1')
            samples += [np.resize(s, (3, 32, 32)) for s in data_dict['data']]
            tasks += [int(fl) for fl in data_dict['coarse_labels']]
            labels += [int(cl) % 5 for cl in data_dict['fine_labels']]
            datasets.append(list(zip(samples, labels, tasks)))
        train_set, test_set = datasets
        return train_set, test_set