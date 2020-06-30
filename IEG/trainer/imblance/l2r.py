import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from thexp import Params, DataBundler, Meter, callbacks, AvgMeter
from datasets.data_loader import ValSet
from arch.metaNets import MetaLeNet
from datasets.data_loader import get_mnist_loader
from .base import Base, BaseParams


class L2R(Base):

    def train_batch(self, eidx, idx, global_step, batch_data, params: BaseParams, device: torch.device):
        meter = Meter()

        (images, labels), (val_images, val_labels) = batch_data

        metanet = MetaLeNet(1).to(device)
        metanet.load_state_dict(self.model.state_dict())
        y_f_hat = metanet(images).squeeze()
        cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduction='none')
        eps = torch.zeros_like(labels, device=device, requires_grad=True)
        l_f_meta = torch.sum(cost * eps)
        metanet.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (metanet.params()), create_graph=True)
        metanet.update_params(params.optim.lr, grads=grads)

        y_g_hat = metanet(val_images).squeeze()
        v_meta_loss = F.binary_cross_entropy_with_logits(y_g_hat, val_labels)
        grad_eps = torch.autograd.grad(v_meta_loss, eps, only_inputs=True)[0]
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        y_f_hat = self.model(images).squeeze()
        cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduction='none')
        l_f = torch.sum(cost * w)

        self.optim.zero_grad()
        l_f.backward()
        self.optim.step()
        meter.l_f = l_f
        meter.meta_l = v_meta_loss
        if (labels == 0).sum() > 0:
            meter.grad_0 = grad_eps[labels == 0].mean()*1e5
            meter.grad_0_max = grad_eps[labels == 0].max()*1e5
            meter.grad_0_min = grad_eps[labels == 0].min()*1e5
        meter.grad_1 = grad_eps[labels == 1].mean()*1e5
        meter.grad_1_max = grad_eps[labels == 1].max()*1e5
        meter.grad_1_min = grad_eps[labels == 1].min()*1e5

        return meter

