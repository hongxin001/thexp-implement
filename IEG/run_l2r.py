from trainer.imblance import L2R
from trainer.imblance.base import BaseParams

params = BaseParams()
params.epoch = 200
params.device = 'cuda:2'
params.from_args()
trainer = L2R(params)
trainer.train()