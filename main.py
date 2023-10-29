from torch import nn

from lightning import Trainer, LightningModule
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from dsta_mvs.model.mvs_model.spherical_sweep_stereo import *
from dsta_mvs.model.feature_extractor import *
from dsta_mvs.model.cost_volume_builder import *
from dsta_mvs.model.cost_volume_regulator import *
from dsta_mvs.model.distance_regressor import *

from dsta_mvs.support.loss_function import *
from dsta_mvs.support.augmentation import *
from dsta_mvs.support.datamodule import *

# Enable use of Tensor Cores
torch.set_float32_matmul_precision('high')

class SaveConfigWandB(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        trainer.logger.log_hyperparams(self.config.as_dict())

def main():
    cli = LightningCLI(
        save_config_callback=SaveConfigWandB,
        save_config_kwargs={
            "config_filename": 'config_backup.yaml',
            "overwrite": True
        }
    )

if __name__ == '__main__':
    main()