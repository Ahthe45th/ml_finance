from transformers import AdamW
import pytorch_lightning as pl
import torch.nn as nn
import transformers
import torch


class BertBaseModel(pl.LightningModule):
    def __init__(
        self,
        model_id: str = None,
        transformer_module: str = None,
        num_labels: int = None,
        **kwargs
    ):
        super().__init__()
        self.kwargs = kwargs
        self.num_labels = num_labels

        self.model_id = model_id
        self.transformer_module = transformer_module

        self.bert = self.bert = eval(
                f"transformers.{transformer_module}.from_pretrained(model_id, num_labels={self.num_labels}, return_dict=True)"
            )
        self.config = self.bert.config


    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask=attention_mask, labels=labels)

        loss = output.loss
        predictions = output.logits

        return loss, predictions

    def shared_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, predictions = self(input_ids, attention_mask, labels)
        self.log("loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": predictions, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        return optimizer
