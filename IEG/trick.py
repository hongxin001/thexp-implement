import torch

def onehot(labels: torch.Tensor,label_num):
    return torch.zeros(labels.shape[0], 10, device=labels.device).scatter_(1, labels.view(-1, 1), 1)

