from trainer.imblance.base import Base
from trainer.imblance.base import BaseParams

params = BaseParams().from_args()
params.epoch = 30
params.device = 'cuda:2'
params.from_args()
params.device = 'cuda:2'
trainer = Base(params)
trainer.train()
