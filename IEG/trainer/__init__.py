from thexp import Params


class GlobalParams(Params):

    def __init__(self):
        super().__init__()
        self.batch_size = 100
        self.max_iteration =200000
        self.learning_rate = 0.1
        self.restore_step =0
        self.dataset = 'cifar10'
        self.ema = True
        self.num_classes = 10
        self.consistency_factor = 20
        self.ce_factor = 5
        self.beta = 0.5  # MixUp hyperparam
        self.nu = 2  # K value for label guessing