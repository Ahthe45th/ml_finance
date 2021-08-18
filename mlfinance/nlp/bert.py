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
N_EPOCHS = 10
BATCH_SIZE = 12
MAX_TOKEN_COUNT = 512
BERT_MODEL_NAME = "bert-base-cased"


# get data
cwd = getpath()
df = pd.read_csv(cwd / "mlfinance" / "nlp" / "toxic_comments.csv")
LABEL_COLUMNS = df.columns.tolist()[2:]


data_module = ToxicCommentDataModule(
    train_df, val_df, tokenizer, batch_size=BATCH_SIZE, max_token_len=MAX_TOKEN_COUNT
)


warmup_steps = 20
total_training_steps = 100

steps_per_epoch = len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS

warmup_steps = total_training_steps // 5
warmup_steps, total_training_steps


model = ToxicCommentTagger(
    n_classes=len(LABEL_COLUMNS),
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps,
)

trainer = pl.Trainer(
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback],
    max_epochs=N_EPOCHS,
    gpus=0,
    progress_bar_refresh_rate=1,
)


trainer.fit(model, data_module)

trainer.save_checkpoint(filepath)