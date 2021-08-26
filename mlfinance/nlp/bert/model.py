"""

Adapted from Venelin Valkov
https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/

"""


from pytorch_lightning.metrics.classification import AUROC
from transformers import AdamW
import pytorch_lightning as pl
import torch.nn as nn
import transformers
import torch


# stops warnings from being seen
transformers.logging.set_verbosity_error()  # comment line when debugging


class BertBaseModel(pl.LightningModule):
    def __init__(
        self,
        model_id: str = None,
        transformer_module: str = None,
        num_labels: int = None,
        labels: list = None,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.num_labels = num_labels
        self.labels = labels

        self.model_id = model_id
        self.transformer_module = transformer_module

        self.bert = self.bert = eval(
            f"transformers.{transformer_module}.from_pretrained(model_id, num_labels={self.num_labels}, return_dict=True)"
        )
        self.config = self.bert.config

        # metrics
        self.sigmoid = (
            nn.Sigmoid()
        )  # need to take sigmoid of logits before calculating AUROC
        self.auroc = AUROC(num_classes=self.num_labels, pos_label=1)
        # AUROC is: https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/
        # The worst AUROC is 0.5, and the best AUROC is 1.0

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = self.bert(input_ids, attention_mask=attention_mask, labels=labels)

        loss = output.loss
        predictions = output.logits

        return loss, predictions

    def shared_step(self, batch, batch_idx, stage):
        loss, predictions = self(batch)

        if stage != "val":
            # only log metrics if batch size is large enough
            if len(batch["labels"]) > 4:
                pred = self.sigmoid(predictions)
                target = batch["labels"].int()  # AUROC only takes integer targets

                for i, name in enumerate(self.labels):
                    try:
                        class_roc_auc = self.auroc(pred[:, i], target[:, i])
                        self.logger.experiment.add_scalar(
                            f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch
                        )
                    except:
                        print(
                            "either batch size not large enough to compute auroch or",
                            " your dataset is not varied enough. computing overall auroch",
                        )
                        class_roc_auc = self.auroc(pred, target)
                        self.logger.experiment.add_scalar(
                            f"overall_roc_auc/Train", class_roc_auc, self.current_epoch
                        )
                        break

        self.log("loss", loss, on_step=True)

        return {"loss": loss, "predictions": predictions, "labels": batch["labels"]}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="fit")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        return optimizer
