import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

# -----------------------------
# 1. Seed Setting
# -----------------------------

def set_seed(seed: int = 42, deterministic: bool = True):
	"""
	Set seed for reproducibility across Python, NumPy, and PyTorch.
	If deterministic=True, also enable deterministic algorithms (may slow down training).
	"""
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	if deterministic:
	    torch.use_deterministic_algorithms(True)
	    torch.backends.cudnn.benchmark = False
	    torch.backends.cudnn.deterministic = True


# -----------------------------
# 2. Device Selection
# -----------------------------

def get_device():
	"""
	Returns the appropriate device string for PyTorch.
	"""
	if torch.cuda.is_available():
	    return torch.device("cuda")
	elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
	    return torch.device("mps")
	else:
	    return torch.device("cpu")

# -----------------------------
# 3. DataLoader Utility
# -----------------------------

# top-level in pytorch_utils.py
def worker_init_fn(worker_id, seed=42):
    import numpy as np
    np.random.seed(seed + worker_id)

def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=4, seed=42,
                   pin_memory=False, persistent_workers=True):
    """
    Returns a PyTorch DataLoader with reproducible shuffling and worker seeds.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    # Use partial instead of lambda
    worker_fn = partial(worker_init_fn, seed=seed)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=worker_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    return loader

# -----------------------------
# 4. Checkpoint Utilities
# -----------------------------

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, seed=None):
    checkpoint = {"model_state": model.state_dict()}

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if seed is not None:
        checkpoint["seed"] = seed

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu", resume=False):
    """
    Load checkpoint into model. If optimizer/scheduler are provided and resume=True, also load their state.
    Returns model, optimizer, scheduler, start_epoch
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    start_epoch = 0
    if optimizer is not None and resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    if scheduler is not None and resume and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return model, optimizer, scheduler, start_epoch

