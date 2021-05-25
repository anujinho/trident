import random

import numpy as np
import torch
from data.taskers import gen_tasks
from PIL.Image import LANCZOS
from torchvision import transforms

from src.zoo.archs import MatchingNetwork


def setup(dataset, root, n_ways, k_shots, q_shots, test_ways, test_shots, test_queries, layers, unrolling_steps, device):
    if dataset == 'omniglot':
        channels = 1
        max_pool = True
        size = 64
        image_trans = transforms.Compose([transforms.Resize(
            28, interpolation=LANCZOS), transforms.ToTensor(), lambda x: 1-x])
        classes = list(range(1623))  # Total classes in Omniglot
        random.shuffle(classes)
        train_tasks = gen_tasks(dataset, root, image_transforms=image_trans,
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, classes=classes[:1100], num_tasks=20000)
        valid_tasks = gen_tasks(dataset, root, image_transforms=image_trans, n_ways=n_ways,
                                k_shots=k_shots, q_shots=q_shots, classes=classes[1100:1200], num_tasks=1024)
        test_tasks = gen_tasks(dataset, root, image_transforms=image_trans,
                               n_ways=test_ways, k_shots=test_shots, q_shots=test_queries, classes=classes[1200:], num_tasks=1024)

    elif dataset == 'miniimagenet':
        channels = 3
        max_pool = True
        size = 1600
        train_tasks = gen_tasks(dataset, root, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        valid_tasks = gen_tasks(dataset, root, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, num_tasks=200)
        test_tasks = gen_tasks(dataset, root, mode='test',
                               n_ways=test_ways, k_shots=test_shots, q_shots=test_queries, num_tasks=200)

    match_net = MatchingNetwork(num_input_channels=channels, stride=(
        2, 2), max_pool=max_pool, lstm_input_size=size, lstm_layers=layers, unrolling_steps=unrolling_steps, device=device)
    return train_tasks, valid_tasks, test_tasks, match_net


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def logits(support, queries, EPSILON):
    # Module with cosine similarity

    n_queries = queries.shape[0]
    n_support = support.shape[0]

    normalised_queries = queries / \
        (queries.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
    normalised_support = support / \
        (support.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

    expanded_x = normalised_queries.unsqueeze(
        1).expand(n_queries, n_support, -1)
    expanded_y = normalised_support.unsqueeze(
        0).expand(n_queries, n_support, -1)

    logits = (expanded_x * expanded_y).sum(dim=2)
    return 1 - logits


def inner_adapt_matching(task, loss, learner, n_ways, k_shots, q_shots, EPSILON, device):
    data, labels = task
    data, labels = data.to(device), labels.to(device)
    total = n_ways * (k_shots + q_shots)
    queries_index = np.zeros(total)

    data = learner.encoder(data)
    # Extracting the evaluation datums from the entire task set, for the meta gradient calculation
    for offset in range(n_ways):
        queries_index[np.random.choice(
            k_shots+q_shots, q_shots, replace=False) + ((k_shots + q_shots)*offset)] = True
    support = data[np.where(queries_index == 0)]
    support_labels = labels[np.where(queries_index == 0)]
    queries = data[np.where(queries_index == 1)]
    queries_labels = labels[np.where(queries_index == 1)]

    support, _, _ = learner.support_encoder(support.unsqueeze(1))
    support = support.squeeze(1)
    queries = learner.query_encoder(queries, support, device)

    preds = logits(queries=queries, support=support, EPSILON=EPSILON)
    attention = (-preds).softmax(dim=1)

    y_onehot = torch.zeros(n_ways * k_shots, n_ways).to(device)

    y = support_labels.unsqueeze(-1)
    y_onehot = y_onehot.scatter(1, y, 1)

    y_pred = torch.mm(attention, y_onehot.to(device))

    # Calculated loss with negative log likelihood
    # Clip predictions for numerical stability
    clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
    eval_loss = loss(clipped_y_pred.log(), queries_labels)
    eval_acc = accuracy(clipped_y_pred, queries_labels)

    return eval_loss, eval_acc
