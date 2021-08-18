#%%
import pandas as pd
import numpy as np

import pytorch_lightning as pl

from getpaths import getpath

from bert.model import ToxicCommentTagger
from bert.data import ToxicCommentDataModule, ToxicCommentsDataset
from callbacks import checkpoint_callback, early_stopping_callback


# get config
RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)


N_EPOCHS = 1
BATCH_SIZE = 1
MAX_TOKEN_COUNT = 512
BERT_MODEL_NAME = "bert-base-cased"


# get data
cwd = getpath()
df = pd.read_csv(cwd / "toxic_comments_small.csv")


data_module = ToxicCommentDataModule(
    df, batch_size=BATCH_SIZE, max_token_len=MAX_TOKEN_COUNT
)


warmup_steps = 1
total_training_steps = 1

warmup_steps = 1
warmup_steps, total_training_steps

model = ToxicCommentTagger(
    n_classes=len(df.columns.tolist()[2:]),
    n_warmup_steps=warmup_steps,
    n_training_steps=1,
)

# need to resize model so that the math works out
model.bert.resize_token_embeddings(len(data_module.tokenizer))


trainer = pl.Trainer(
    checkpoint_callback=checkpoint_callback,
    max_epochs=N_EPOCHS,
    gpus=0,
    progress_bar_refresh_rate=1,
)


trainer.fit(model, data_module)

trainer.save_checkpoint("test_checkpoint")
