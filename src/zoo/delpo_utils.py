import random

import learn2learn as l2l
import numpy as np
import torch
from torch.nn import functional as F
from data.taskers import gen_tasks
from PIL.Image import LANCZOS
from torchvision import transforms

from src.zoo.archs import CCVAE


def setup(dataset, root, n_ways, k_shots, q_shots, order, inner_lr, device, download):
    if dataset == 'omniglot':
        image_trans = transforms.Compose([transforms.Resize(
            28, interpolation=LANCZOS), transforms.ToTensor(), lambda x: 1-x])
        classes = list(range(1623))  # Total classes in Omniglot
        random.shuffle(classes)
        # Generating tasks and model according to the MAML implementation for Omniglot
        train_tasks = gen_tasks(dataset, root, image_transforms=image_trans,
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, classes=classes[:1100], num_tasks=20000)
        valid_tasks = gen_tasks(dataset, root, image_transforms=image_trans, n_ways=n_ways,
                                k_shots=k_shots, q_shots=q_shots, classes=classes[1100:1200], num_tasks=200)
        test_tasks = gen_tasks(dataset, root, image_transforms=image_trans,
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, classes=classes[1200:], num_tasks=600)
        learner = CCVAE(in_channels=1, base_channels=64, n_ways=n_ways, dataset='omniglot')

    elif dataset == 'miniimagenet':
        # Generating tasks and model according to the MAML implementation for MiniImageNet
        train_tasks = gen_tasks(dataset, root, download=download, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        valid_tasks = gen_tasks(dataset, root, download=download, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        test_tasks = gen_tasks(dataset, root, download=download, mode='test',
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, num_tasks=600)
        learner = CCVAE(in_channels=3, base_channels=64, n_ways=n_ways, dataset='mini_imagenet')

    learner = learner.to(device)
    learner = l2l.algorithms.MAML(learner, first_order=order, lr=inner_lr)

    return train_tasks, valid_tasks, test_tasks, learner


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def kl_div(mus, log_vars):
    return - 0.5 * (1 + log_vars - mus**2 - torch.exp(log_vars)).sum(dim=1)

def loss(reconst_loss: object, reconst_image, image, logits, labels, mu_s, log_var_s, mu_l, log_var_l):
    kl_div_s = kl_div(mu_s, log_var_s).mean()
    kl_div_l = kl_div(mu_l, log_var_l).mean()

    ce_loss = torch.nn.CrossEntropyLoss()
    classification_loss = ce_loss(F.softmax(logits, dim=1), labels)
    rec_loss = reconst_loss(reconst_image, image)

    L = classification_loss + rec_loss + kl_div_s + kl_div_l  # -log p(x,y)
    return L 

def inner_adapt_delpo(task, reconst_loss, learner, n_ways, k_shots, q_shots, adapt_steps, device):
    data, labels = task
    data, labels = data.to(device), labels.to(device)
    total = n_ways * (k_shots + q_shots)
    queries_index = np.zeros(total)

    # Extracting the evaluation datums from the entire task set, for the meta gradient calculation
    for offset in range(n_ways):
        queries_index[np.random.choice(
            k_shots+q_shots, q_shots, replace=False) + ((k_shots + q_shots)*offset)] = True
    support = data[np.where(queries_index == 0)]
    support_labels = labels[np.where(queries_index == 0)]
    queries = data[np.where(queries_index == 1)]
    queries_labels = labels[np.where(queries_index == 1)]

    # Inner adapt step
    for _ in range(adapt_steps):
        reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner(support)
        adapt_loss = loss(reconst_loss, reconst_image, support, logits, support_labels, mu_s, log_var_s, mu_l, log_var_l)
        learner.adapt(adapt_loss)

    reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner(queries)
    eval_loss = loss(reconst_loss, reconst_image, queries, logits, queries_labels, mu_s, log_var_s, mu_l, log_var_l)
    eval_acc = accuracy(F.softmax(logits, dim=1), queries_labels)
    return eval_loss, eval_acc
