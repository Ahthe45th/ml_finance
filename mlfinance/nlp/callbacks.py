from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def checkpoint_callback():
    callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )


def early_stopping_callback():
    callback = EarlyStopping(monitor="val_loss", patience=2)
    return callback
