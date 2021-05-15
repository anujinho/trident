import random

import learn2learn as l2l
import numpy as np
from data.taskers import gen_tasks
from PIL.Image import LANCZOS
from torchvision import transforms

from src.zoo.archs import MiniImageCNN, OmniCNN


def setup(dataset, root, n_ways, k_shots, q_shots, order, inner_lr, device):
    if dataset == 'omniglot':
        image_trans = transforms.Compose([transforms.Resize(
            28, interpolation=LANCZOS), transforms.ToTensor(), lambda x: 1-x])
        classes = list(range(1623))  # Total classes in Omniglot
        random.shuffle(classes)
        # Generating tasks and model according to the MAML implementation for Omniglot
        train_tasks = gen_tasks(dataset, root, image_transforms=image_trans,
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, classes=classes[:1100], num_tasks=20000)
        valid_tasks = gen_tasks(dataset, root, image_transforms=image_trans, n_ways=n_ways,
                                k_shots=k_shots, q_shots=q_shots, classes=classes[1100:1200], num_tasks=600)
        test_tasks = gen_tasks(dataset, root, image_transforms=image_trans,
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, classes=classes[1200:], num_tasks=600)
        learner = OmniCNN(output_size=n_ways, stride=(2, 2))

    elif dataset == 'miniimagenet':
        # Generating tasks and model according to the MAML implementation for MiniImageNet
        train_tasks = gen_tasks(dataset, root, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        valid_tasks = gen_tasks(dataset, root, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        test_tasks = gen_tasks(dataset, root, mode='test',
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        learner = MiniImageCNN(output_size=n_ways, stride=(2, 2))

    learner = learner.to(device)
    learner = l2l.algorithms.MAML(learner, first_order=order, lr=inner_lr)

    return train_tasks, valid_tasks, test_tasks, learner


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def inner_adapt_maml(task, loss, learner, n_ways, k_shots, q_shots, adapt_steps, device):
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
        adapt_loss = loss(learner(support), support_labels)
        learner.adapt(adapt_loss)

    preds = learner(queries)
    eval_loss = loss(preds, queries_labels)
    eval_acc = accuracy(preds, queries_labels)
    return eval_loss, eval_acc
