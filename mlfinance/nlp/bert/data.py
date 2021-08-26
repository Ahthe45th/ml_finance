"""

Adapted from Venelin Valkov
https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/

"""


import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch import FloatTensor
from getpaths import getpath
from typing import Optional
from os.path import abspath
import pandas as pd


class BertDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_token_len: int = 128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.labels = self.data.columns.tolist()[2:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        comment_text = data_row.comment_text
        labels = data_row[self.labels]

        encoding = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            comment_text=comment_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=FloatTensor(labels),
        )


class BertDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_id=None,
        data_path=None,
        tokenizer=None,
        batch_size=1,
        max_token_len=128,
        num_workers=0,
    ):

        super().__init__()

        if model_id == None:
            self.model_id = "bert-base-uncased"
        else:
            self.model_id = model_id

        if data_path == None:
            cwd = getpath(abspath(__file__), custom=True)
            self.df = pd.read_csv(cwd / ".." / ".." / "toxic_comments_small.csv")
        else:
            self.df = pd.read_csv(data_path)

        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        self.labels = self.df.columns.tolist()[2:]
        self.num_labels = len(self.labels)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = eval(f"transformers.{self.tokenizer}")
        self.tokenizer = self.tokenizer.from_pretrained(self.model_id)

        if stage == "fit" or stage == None:
            train_df, val_df = train_test_split(self.df, test_size=0.2)

            self.train_dataset = BertDataset(
                train_df, self.tokenizer, self.max_token_len
            )

            self.val_dataset = BertDataset(val_df, self.tokenizer, self.max_token_len)

        elif stage == "test":
            self.test_dataset = BertDataset(self.df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    # TODO: write short example of how to use DataModule
    pass
