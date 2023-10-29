
import torch
from lightning.pytorch.callbacks import Callback

class OfflineSingleGPUValidation(Callback):
    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug
        self.metrics = dict()

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def on_validation_start(self, trainer, pl_module ):
        # Populate all the metrics according to the validation datasets.
        self.metrics = dict()
        # Debug use.
        self._debug_print(f'OfflineSingleGPUValidation: on_validation_start. ')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        '''Assuming that the outputs is a dictionary of metrics.
        Metrics only contain values. No images involved.
        '''
        
        # Update the metrics.
        for k, v in outputs.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the logger.
        logger = pl_module.logger.experiment

        # Compute the mean of the metrics.
        epoch_metrics = dict()
        for metric_name, metric_value in self.metrics.items():
            # Compose the new metric name.
            new_metric_name = f'{metric_name}_epoch'
            epoch_metrics[new_metric_name] = torch.stack(metric_value).mean()

        # Log the metrics.
        logger.log( epoch_metrics )
        self._debug_print(f'OfflineSingleGPUValidation: on_validation_epoch_end. ')
