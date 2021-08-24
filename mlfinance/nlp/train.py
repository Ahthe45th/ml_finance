"""

Adapted from Venelin Valkov
https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/


"""


import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from getpaths import getpath


try:
    from .utils import using_gpu
except ImportError:
    from utils import using_gpu


def main(cfg):
    """
    training, testing, and evaluating done here
    """
    if cfg.mode == "train":

        if using_gpu():
            cfg = self.cfg["gpu"]
        else:
            cfg = self.cfg["cpu"]

        bert = Bert(cfg)
        bert.train()

    elif cfg.mode == "test":
        pass
    else:
        pass


class Bert:
    def __init__(self, cfg=None, overrides=[]):
        # do configuration with overrides
        if cfg == None:
            if using_gpu():
                config_path = "conf/gpu"
            else:
                config_path = "conf/cpu"

            with initialize(config_path=config_path):
                self.cfg = compose(config_name="default", overrides=overrides)
        else:
            self.cfg = cfg

        self.model = None
        self.datamodule = None
        self.trainer = None

    def train(self):
        if self.cfg.custom == False:
            # show configs at start of training session
            print(OmegaConf.to_yaml(self.cfg))

            self.datamodule = hydra.utils.instantiate(self.cfg.datamodule)

            # make sure that the model is the right size
            self.cfg.model.num_labels = self.datamodule.num_labels

            self.model = hydra.utils.instantiate(self.cfg.model)

            self.trainer = hydra.utils.instantiate(self.cfg.trainer)

            # make sure the tokenizer is for the right model
            self.datamodule.model_id = self.model.model_id

        self.trainer.fit(self.model, self.datamodule)

        self.trainer.save_checkpoint(self.cfg.checkpoint.name)


@hydra.main(config_path="./conf", config_name="config")
def cli_hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    cli_hydra_entry()
