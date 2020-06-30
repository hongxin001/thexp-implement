from torch.optim import SGD
from torch import autograd
import torch
from torch import nn
from thexp import RndManager



rnd = RndManager()
lr = 0.1
rnd.mark('temp')
w = nn.Linear(1,1,bias=False)
w2 = nn.Linear(1,1,bias=False)
# print('backward')
# print(w.weight.data,w2.weight.data)
model = nn.Sequential(w,w2)
optim = SGD(model.parameters(),lr=lr,momentum=False)

#
x = torch.Tensor([1.])
y = torch.Tensor([5.])

# for _ in range(500):
#     y_h = model(x)
#
#     loss = (y-y_h)**2
#     optim.zero_grad()
#     loss.backward(create_graph=True)
#     optim.step()
#     # print([param.grad for param in model.parameters()])
# print(loss,y_h,w.weight.data,w2.weight.data)

print(model._parameters)
print(model._modules)



print(list(model._named_members(lambda module: module._parameters.items(),prefix='kk')))
