import random

import numpy as np
import torch
from data.taskers import gen_tasks
from PIL.Image import LANCZOS
from torchvision import transforms
from torch.utils.data import DataLoader

from src.zoo.archs import EncoderNN


def setup(dataset, root, n_ways, k_shots, q_shots, test_ways, test_shots, test_queries, device):
    if dataset == 'omniglot':
        channels = 1
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
                               n_ways=test_ways, k_shots=test_shots, q_shots=test_queries, classes=classes[1200:], num_tasks=200)
        
    elif dataset == 'miniimagenet':
        channels = 3
        # Generating tasks and model according to the MAML implementation for MiniImageNet
        train_tasks = gen_tasks(dataset, root, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        valid_tasks = gen_tasks(dataset, root, mode='validation',
                                n_ways=test_ways, k_shots=test_shots, q_shots=test_queries, num_tasks=200)
        test_tasks = gen_tasks(dataset, root, mode='test',
                               n_ways=test_ways, k_shots=test_shots, q_shots=test_queries, num_tasks=200)
        
#     train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)
#     valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)
#     test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)
        
    learner = EncoderNN(channels=channels, max_pool=True, stride=(2,2))
    learner = learner.to(device)
    
    return train_tasks, valid_tasks, test_tasks, learner


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def logits(support, queries, n, k, q):
    prototypes = support.view(n, k, -1).mean(dim=1)
    a = queries.shape[0]
    b = prototypes.shape[0]
    logits = -((queries.unsqueeze(1).expand(a,b,-1) - prototypes.unsqueeze(0).expand(a,b,-1))**2).sum(dim=2)
    return logits


def inner_adapt_proto(task, loss, learner, n_ways, k_shots, q_shots, device):
    data, labels = task
    data, labels = data.to(device), labels.to(device)
    #data, labels = data.squeeze(0), labels.squeeze(0)
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)
    total = n_ways * (k_shots + q_shots)
    queries_index = np.zeros(total)

    data = learner(data)
    # Extracting the evaluation datums from the entire task set, for the meta gradient calculation
    for offset in range(n_ways):
        queries_index[np.random.choice(
            k_shots+q_shots, q_shots, replace=False) + ((k_shots + q_shots)*offset)] = True
    support = data[np.where(queries_index == 0)]
    #support_labels = labels[np.where(queries_index == 0)]
    queries = data[np.where(queries_index == 1)]
    queries_labels = labels[np.where(queries_index == 1)]

    preds = logits(queries=queries, support=support, n=n_ways, k=k_shots, q=q_shots)
    eval_loss = loss(preds, queries_labels.long())
    eval_acc = accuracy(preds, queries_labels)

    return eval_loss, eval_acc
