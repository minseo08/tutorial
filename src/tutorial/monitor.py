from dotmap import DotMap


class Monitor:
    """Dummy class for monitoring. Accepts inputs but performs no operations."""
    _instance = None

    def __new__(cls, args=None):
        if cls._instance is None:
            if args is None:
                raise ValueError("Monitor instance cannot be created without args.")

            cls._instance = super(Monitor, cls).__new__(cls)
            cls._instance._initialize(args)
        return cls._instance

    def _initialize(self, args):
        self.args = args
        self._init_wandb()
        self.set_hyperparams()

    def _init_wandb(self):
        pass

    def set_hyperparams(self):
        config = self.args
        self.hyperparams = DotMap({
            "model": config.model,
            "embedding_dim": config.embedding_dim,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "metric": config.metric,
            "data_path": config.data_path,
            "train_ratio": config.train_ratio,
            "valid_ratio": config.valid_ratio
        })

    def get_hyperparams(self):
        return self.hyperparams

    def log(self, log_data: dict):
        pass

    def finish(self):
        pass