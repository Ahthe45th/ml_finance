"""

this railroad track hauls callbacks that are available for training

"""

from pytorch_lightning.callbacks import ModelPruning, ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_debug
import pytorch_lightning as pl
from typing import Dict, Any
import torch
import os


class CustomModelPruning(ModelPruning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_save_checkpoint(
        self, trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]
    ):
        if self._make_pruning_permanent:
            print(
                "\nsaving pruned model to ./lightning_logs/version_num/checkpoints...\n"
            )

            rank_zero_debug(
                "`ModelPruning.on_save_checkpoint`. Pruning is made permanent for this checkpoint."
            )
            prev_device = pl_module.device
            # prune a copy so training can continue with the same buffers
            # the copy has to be saved to disk temporarily though
            weights_path = "weights_temp.pt"
            torch.save(pl_module.to("cpu"), weights_path)
            copy = torch.load(weights_path)
            # delete weights saved on disk
            os.remove(weights_path)
            # continue pruning the model
            self.make_pruning_permanent(copy)
            checkpoint["state_dict"] = copy.state_dict()
            pl_module.to(prev_device)

        # garbage collection
        del prev_device, copy
        # empty cache
        torch.cuda.empty_cache()
