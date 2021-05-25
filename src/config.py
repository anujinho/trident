maml_omniglot = {
    'root': '/home/anuj/Desktop/Work/TU_Delft/research/implement/omniglot',
    'n_ways': 5,
    'k_shots': 1,
    'q_shots': 1,
    'inner_adapt_steps': 1,
    'inner_lr': 0.5,
    'meta_lr': 0.003,
    'meta_batch_size': 32,
    'iterations': 60000,
    'order': False,
    'device': 'cuda'
}

maml_mini = {
    'root': '/home/anuj/Desktop/Work/TU_Delft/research/implement/mini_imagenet',
    'n_ways': 5,
    'k_shots': 1,
    'q_shots': 1,
    'inner_adapt_steps': 1,
    'inner_lr': 0.5,
    'meta_lr': 0.003,
    'meta_batch_size': 32,
    'iterations': 60000,
    'order': False,
    'device': 'cuda'
}

proto_omni = {
    'root': '/home/anuj/Desktop/Work/TU_Delft/research/implement/omniglot',
    'n_ways': 60,
    'k_shots': 1,
    'q_shots': 5,
    'test_ways': 5,
    'test_shots': 1,
    'test_queries': 5,
    'lr': 0.001,
    'meta_batch_size': 100,
    'iterations': 250,
    'device': 'cuda'
}

proto_mini = {
    'root': '/home/anuj/Desktop/Work/TU_Delft/research/implement/mini_imagenet',
    'n_ways': 30,
    'k_shots': 1,
    'q_shots': 15,
    'test_ways': 5,
    'test_shots': 1,
    'test_queries': 30,
    'lr': 0.001,
    'meta_batch_size': 100,
    'iterations': 250,
    'device': 'cuda'
}

matching_omni = {
    'root': '/home/anuj/Desktop/Work/TU_Delft/research/implement/omniglot',
    'n_ways': 5,
    'k_shots': 1,
    'q_shots': 15,
    'test_ways': 5,
    'test_shots': 1,
    'test_queries': 1,
    'lr': 0.001,
    'meta_batch_size': 100,
    'iterations': 100,
    'layers': 1,
    'unrolling_steps': 2,
    'device': 'cuda'
}

matching_mini = {
    'root': '/home/anuj/Desktop/Work/TU_Delft/research/implement/mini_imagenet',
    'n_ways': 5,
    'k_shots': 1,
    'q_shots': 15,
    'test_ways': 5,
    'test_shots': 1,
    'test_queries': 1,
    'lr': 0.001,
    'meta_batch_size': 100,
    'iterations': 200,
    'layers': 1,
    'unrolling_steps': 2,
    'device': 'cuda'
}