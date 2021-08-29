"""

Adapted from Venelin Valkov
https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/


"""
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from getpaths import getpath
import sys, os

try:
    from .utils import using_gpu
except ImportError:
    from utils import using_gpu

def main(cfg):
    """
    training, testing, and evaluating done here
    """
    if cfg.mode == "train":
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
        self.callbacks = []

        self.model_pruning = False

    def pre_loop(self):
        pass

    def initialize(self, user_triggered=True):

        if self.cfg.custom == False:
            # show configs at start of training session
            print(OmegaConf.to_yaml(self.cfg))

            self.datamodule = hydra.utils.instantiate(self.cfg.datamodule)

            # make sure that the model is the right size
            self.cfg.model.num_labels = self.datamodule.num_labels
            self.cfg.model.labels = self.datamodule.labels

            self.model = hydra.utils.instantiate(self.cfg.model)

            self.trainer = hydra.utils.instantiate(self.cfg.trainer)

            # make sure the tokenizer is for the right model
            self.datamodule.model_id = self.model.model_id

        if user_triggered == True:
            # if user initializes modules on their own
            # it will initialize with cfg once, then never again
            # except if user_triggered = False
            self.cfg.custom = True
            print("custom module swapping now enabled")

    def load_from_checkpoint(self, checkpoint_path: str = None):
        # load checkpoint
        if checkpoint_path != None:
            self.cfg.model.checkpoint_path = checkpoint_path

        if self.cfg.model.checkpoint_path != None:
            # this is a hack, but will be useful if we ever do transfer learning in the future
            # this ONLY loads weights that are named the same way in the old and new model
            old_state_dict = torch.load(self.cfg.model.checkpoint_path)["state_dict"]
            for key in old_state_dict.keys():
                try:
                    keys = key.split(".")[:-1]
                    module_name = (
                        "model._modules['"
                        + "']._modules['".join(keys)
                        + "'].weight.data"
                    )
                    setattr(self, module_name, old_state_dict[key])
                except Exception as e:
                    print(e, f"could not find {key} in self.model")

    def load_checkpoint(self, checkpoint_path: str = None):
        self.load_from_checkpoint(checkpoint_path)

    def load_callbacks(self):
        # monitor loss, and save checkpoint when loss gets better
        self.trainer.callbacks.append(
            ModelCheckpoint(
                monitor="loss",
                filename=f"{self.cfg.checkpoint.name}",
                save_top_k=1,
                mode="min",
                every_n_val_epochs=0,
                save_weights_only=True,
            )
        )

        # prune model for faster model inference
        if self.model_pruning == True:
            self.trainer.callbacks.append(
                CustomModelPruning("l1_unstructured", amount=0.5)
            )

        for callback in self.callbacks:
            self.trainer.callbacks.append(callback)

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

        self.initialize(user_triggered=False)

        self.load_from_checkpoint()

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
