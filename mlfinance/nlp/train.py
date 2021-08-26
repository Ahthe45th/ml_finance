"""

Adapted from Venelin Valkov
https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/


"""


from ml_finance.mlfinance.nlp.callbacks import CustomModelPruning, ModelCheckpoint
from ml_finance.mlfinance.nlp.utils import using_gpu
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from getpaths import getpath
import warnings
import hydra
import torch


# stops warnings from being seen
warnings.simplefilter("ignore")  # comment line when debugging


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

    def pre_loop(self):
        pass

    def initialize(self):
        if self.cfg.custom == False:
            # show configs at start of training session
            print(OmegaConf.to_yaml(self.cfg))

            self.datamodule = hydra.utils.instantiate(self.cfg.datamodule)

            # make sure that the model is the right size
            self.cfg.model.num_labels = self.datamodule.num_labels
            self.cfg.model.labels = self.datamodule.labels

            self.model = hydra.utils.instantiate(self.cfg.model)

            # load checkpoint
            if self.cfg.model.checkpoint_path != None:
                self.cfg.model.load_checkpoint(self.cfg.model.checkpoint_path)

            self.trainer = hydra.utils.instantiate(self.cfg.trainer)

            # make sure the tokenizer is for the right model
            self.datamodule.model_id = self.model.model_id

    def load_callbacks(self):
        # monitor loss, and save checkpoint when loss gets better
        self.trainer.callbacks.append(
            ModelCheckpoint(
                monitor="loss",
                filename=f"{self.cfg.checkpoint.name}",
                save_top_k=1,
                mode="min",
                every_n_val_epochs=0,
            )
        )

        # prune model for faster model inference
        self.trainer.callbacks.append(CustomModelPruning("l1_unstructured", amount=0.5))

    def quantize_model(self):
        # make smaller, quantized model
        print("\nsaving ONNX model")
        self.model.eval()
        example_data = next(iter(self.datamodule.train_dataloader()))

        torch.onnx.export(
            self.model,  # model being run
            example_data,  # model input (or a tuple for multiple inputs)
            f"{self.cfg.checkpoint.name}.onnx",  # where to save the model
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["modelInput"],  # the model's input names
            output_names=["modelOutput"],  # the model's output names
            dynamic_axes={
                "modelInput": {0: "batch_size"},  # variable length axes
                "modelOutput": {0: "batch_size"},
            },
        )

    def post_loop(self):
        pass

    def train(self):
        self.pre_loop()

        self.initialize()

        self.load_callbacks()

        self.trainer.fit(self.model, self.datamodule)

        self.quantize_model()

        self.post_loop()


class DistilBert(Bert):
    def __init__(self, cfg=None, overrides=[]):
        super().__init__(cfg=cfg, overrides=overrides)

    def pre_loop(self):
        self.cfg.model.model_id = "distilbert-base-uncased"
        self.cfg.model.transformer_module = "DistilBertForSequenceClassification"
        self.cfg.datamodule.tokenizer = "DistilBertTokenizerFast"


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
