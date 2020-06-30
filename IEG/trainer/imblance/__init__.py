from thexp import Params


class ImblanceParams(Params):

    def __init__(self):
        super().__init__()
        self.optim = {
            'lr': 1e-3,
            'momentum': 0.9,
        }
        self.batch_size = 100
        self.classes = [9, 4]
        self.train_proportion = 0.995