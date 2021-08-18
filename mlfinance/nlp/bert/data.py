from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd



class ToxicCommentsDataset(Dataset):
    def __init__(
        self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        comment_text = data_row.comment_text
        labels = data_row[LABEL_COLUMNS]

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
            labels=torch.FloatTensor(labels),
        )

class ToxicCommentDataModule(pl.LightningDataModule):
    def __init__(self, df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = None
        self.test_df = None
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len


    def setup(self, stage=None):
        self.train_df, self.test_df = train_test_split(df, test_size=0.05)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )


    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=0
        )


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=0
        )


if __name__ == "__main__":
    #TODO: write short example of how to use DataModule
    pass