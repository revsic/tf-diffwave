from dataset.config import Config as DataConfig
from model.config import Config as ModelConfig


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self):
        # optimizer
        self.learning_rate = 2e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        # 13000:100
        self.split = 13000
        self.bufsiz = 48

        self.epoch = 10000

        # path config
        self.log = './log'
        self.ckpt = './ckpt'

        # model name
        self.name = 'b1'

        # interval configuration
        self.eval_intval = 5000
        self.ckpt_intval = 10000

class Config:
    """Integrated configuration.
    """
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.train = TrainConfig()

    def dump(self):
        """Dump configurations into serializable dictionary.
        """
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration.
        """
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes.
    """
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
