import torch


def get_device() -> torch.device:
    """Usa a GPU disponível (Apple MPS ou CUDA) e cai para CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
