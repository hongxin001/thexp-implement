from IEG.arch.onelayernet import MetaTNet
from torch import nn
from thexp import RndManager
rnd = RndManager()

rnd.mark("temp")
a = MetaTNet()
b = MetaTNet()

# for k in a.name_params(True):
#     print(k)


for k,v,module in a.name_params(with_module=True):
    print(k,type(v),type(module))
    if isinstance(v,nn.Parameter):
        print(k,v)