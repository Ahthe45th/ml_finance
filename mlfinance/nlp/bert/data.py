from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor
from transformers import BertTokenizerFast as BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pytorch_lightning as pl
from typing import Optional
from getpaths import getpath



class ToxicCommentsDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_token_len: int = 128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.label_columns = self.data.columns.tolist()[2:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        comment_text = data_row.comment_text
        labels = data_row[self.label_columns]

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



class ToxicCommentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name = None,
        data_path=None,
        batch_size=1,
        max_token_len=128,
        num_workers=0
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.model_name = model_name
        if self.model_name == None:
            self.model_name = "bert-base-uncased"
        
        if data_path == None:
            df = pd.read_csv(getpath()/'toxic_comments_small.csv')
        else:
            df = pd.read_csv(data_path)

        self.n_classes = len(df.columns.tolist()[2:])
        self.train_df, self.test_df = train_test_split(df, test_size=0.2)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.max_token_len = max_token_len

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ToxicCommentsDataset(
            self.train_df, self.tokenizer, self.max_token_len
        )

        self.test_dataset = ToxicCommentsDataset(
            self.test_df, self.tokenizer, self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)



if __name__ == "__main__":
    # TODO: write short example of how to use DataModule
    pass
