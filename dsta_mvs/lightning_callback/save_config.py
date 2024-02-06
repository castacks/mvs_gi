
from lightning import Trainer, LightningModule
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback

class SaveConfigWandB(SaveConfigCallback):
    # def setup(self, trainer: Trainer, 
    #           pl_module: LightningModule, # pl for "PyTorch Lightning".
    #           stage: str) -> None:
    #     super().setup(trainer, pl_module, stage)

    #     config_dict = self.config.as_dict()
    #     trainer.logger.log_hyperparams(config_dict)
        
    def save_config(self, 
                    trainer: Trainer, 
                    pl_module: LightningModule, 
                    stage: str) -> None:
        config_dict = self.config.as_dict()
        trainer.logger.log_hyperparams(config_dict)
