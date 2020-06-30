import torch
from thexp import Trainer, callbacks, Params, DataBundler, Meter, AvgMeter
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from datasets.mnist_imblance import get_mnist_loader, ValSet
from trainer.imblance import ImblanceParams


class Base(Trainer, callbacks.TrainCallback):
    def callbacks(self, params: ImblanceParams):
        super().callbacks(params)
        callbacks.LoggerCallback().hook(self)
        callbacks.EvalCallback(1, 1).hook(self)

    def datasets(self, params: ImblanceParams):
        super().datasets(params)
        data_loader = get_mnist_loader(params.batch_size, classes=params.classes, proportion=params.train_proportion, mode="train")
        test_loader = get_mnist_loader(params.batch_size, classes=params.classes, proportion=0.5, mode="test")
        val_dataset = ValSet(data_loader.dataset.data_val, data_loader.dataset.labels_val)
        val_loader = DataLoader(val_dataset, params.batch_size, drop_last=False, shuffle=True)
        train_loader = DataBundler().add(data_loader).cycle(val_loader).zip_mode()
        self.regist_databundler(
            train=train_loader,
            test=test_loader)
        self.to(self.device)

    def models(self, params: ImblanceParams):
        from arch.lenet import LeNet
        self.model = LeNet(1)  # type:nn.Module
        self.optim = SGD(self.model.parameters(), **params.optim)
        self.to(self.device)

    def train_batch(self, eidx, idx, global_step, batch_data, params: ImblanceParams, device: torch.device):
        meter = Meter()
        (images, labels), (_, _) = batch_data

        logits = self.model(images).squeeze()
        meter.ce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.optim.zero_grad()
        meter.ce_loss.backward()
        self.optim.step()
        return meter

    def test_eval_logic(self, dataloader, param: Params):
        meter = AvgMeter()
        for itr, (images, labels) in enumerate(dataloader):
            output = self.model(images).squeeze()
            predicted = (torch.sigmoid(output) > 0.5).int()
            meter.acc = (predicted.int() == labels.int()).float().mean().detach()

        return meter
