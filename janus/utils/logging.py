# WandBLogger: experiment tracking with Weights & Biases
import wandb

class WandBLogger:
    def __init__(self, project, config):
        self.run = wandb.init(project=project, config=config)

    def log(self, data):
        self.run.log(data)
