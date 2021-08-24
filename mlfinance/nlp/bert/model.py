from transformers import get_linear_schedule_with_warmup, AdamW
import pytorch_lightning as pl
import torch.nn as nn
import transformers
import torch


class BertBaseModel(pl.LightningModule):
    def __init__(
        self,
        n_classes: int = None,
        n_training_steps: int = None,
        n_warmup_steps: int = None,
        model_id: str = "bert-base-cased",
        **kwargs
    ):
        super().__init__()
        self.kwargs = kwargs
        
        self.model_id = model_id
        if 'transformer_module' in kwargs:
            self.bert = eval(f"transformers.{self.kwargs['transformer_module']}.from_pretrained(model_id, return_dict=True)")
        else:
            self.bert = transformers.BertModel.from_pretrained(model_id, return_dict=True)

        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        try:
            output = self.classifier(output.pooler_output)
        except:
            # Right way to get hidden state from DistilBert
            # https://huggingface.co/transformers/_modules/transformers/modeling_distilbert.html
            hidden_state = output[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            # pooled_output = self.dropout(pooled_output)  # (bs, dim)
            output = self.classifier(pooled_output)

        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )
