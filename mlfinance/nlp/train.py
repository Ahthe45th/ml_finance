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
    
    def pre_train_loop(self):
        pass

    def post_train_loop(self):
        pass

    def train(self):
        self.pre_train_loop()

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

        self.post_train_loop()


class DistilBert(Bert):
    def __init__(self, cfg=None, overrides=[]):
        super().__init__(cfg=cfg, overrides=overrides)
    
    def pre_train_loop(self):
        self.cfg.model.model_id='distilbert-base-uncased'
        self.cfg.model.transformer_module='DistilBertForSequenceClassification'
        self.cfg.datamodule.tokenizer='DistilBertTokenizerFast'


def main(cfg):
    """
    training, testing, and evaluating
    using CLI workflow done here
    """
    if cfg.mode == "train":

        if using_gpu():
            cfg = cfg["gpu"]
        else:
            cfg = cfg["cpu"]

        bert = Bert(cfg)
        bert.train()

    elif cfg.mode == "test":
        pass
    else:
        pass


@hydra.main(config_path="./conf", config_name="config")
def cli_hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    cli_hydra_entry()
