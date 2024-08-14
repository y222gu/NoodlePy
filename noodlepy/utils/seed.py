import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import numpy as np
import torch
import pytorch_lightning as pl


def seed_all_random_process(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    pl.seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    