import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
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
        pass
    elif cfg.mode == "test":
        pass
    else:
        pass





class Bert():
    def __init__(self):
        # get config file
        initialize(config_path="conf")
        self.cfg = compose(config_name="config")

        if using_gpu():
            self.cfg = self.cfg['gpu']
        else:
            self.cfg = self.cfg['cpu']
        

    def train(self):
        print('loading model from https://huggingface.co/api/models/...') #this takes some time
        self.model = hydra.utils.instantiate(self.cfg.model)

        self.datamodule = hydra.utils.instantiate(self.cfg.datamodule)

        self.trainer = hydra.utils.instantiate(self.cfg.trainer)

        # make sure the tokenizer is for the right model
        self.datamodule.name = self.model.name
        # make sure that the model is the right size
        self.model.n_classes = self.datamodule.n_classes
        self.model.bert.resize_token_embeddings(len(self.datamodule.tokenizer))

        self.trainer.fit(self.model, self.datamodule)

        self.trainer.save_checkpoint(self.cfg.checkpoint.name)


@hydra.main(config_path="./conf", config_name="config")
def cli_hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    cli_hydra_entry()
