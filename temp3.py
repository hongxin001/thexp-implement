from IEG.arch.metaNets import MetaLeNet,LeNet

from torch.nn import functional as F
import torch
from torch import autograd
from torchvision.datasets.fakedata import FakeData
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader

dataset = FakeData(50, image_size=(28, 28), transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=2)

metanet = MetaLeNet()

lenet = LeNet()

# print(next(metanet.params())[0])
# metanet.load_state_dict(lenet.state_dict())
# print(next(metanet.params())[0])
# print(metanet.state_dict()['conv1.weight'])
# print(lenet.state_dict()['conv1.weight'])

dataiter = iter(dataloader)
x, y = next(dataiter)
y_h = metanet(x)
print(x.shape, y.shape, y_h.shape)

raw_loss = F.cross_entropy(y_h, y, reduction='none')
eps = autograd.Variable(torch.zeros(x.shape[0]),requires_grad = True)

loss = torch.sum(eps * raw_loss)

metanet.zero_grad()
grads = autograd.grad(loss, (metanet.params()), create_graph=True,allow_unused=True)
metanet.update_params(0.1,grads)

x_val, y_val = next(dataiter)

y_g_hat = metanet(x_val)
raw_loss_val = F.cross_entropy(y_g_hat,y_val)

grad_eps =autograd.grad(raw_loss_val,eps)

print(grad_eps)
