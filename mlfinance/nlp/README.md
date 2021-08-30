<table align="center"><tr><td align="center" width="9999">

# 🔥 nlp module 🔥




<p align="center">
  <img src="https://raw.githubusercontent.com/facebookresearch/hydra/master/website/static/img/Hydra-Readme-logo2.svg" width="200" />
  <img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="60" /> 
</p>



</td></tr><tr align="left"></tr></table>



Ever wanted to train an NLP model to classify texts? Worried that your model will only work on gpu? Then you're in the right place.

<br><br>

Training a text classifier is easy:

```python
from mlfinance.nlp import Bert, DistilBert

bert = Bert() # train Bert on toy dataset
bert.train()

distilbert = DistilBert() # train DistilBert on the same dataset
distilbert.train()

# training will automatically save onnx files into your current working directory
```

<br><br>

In order to view the how the model is learning while it's training, run this on the command line:

> tensorboard --logdir lightning_logs

<br><br>

---

## Training

Now you're probably wanting to train on your own dataset, not the toy dataset that this module shipped with. That's also easy!

```python

from mlfinance.nlp import Bert

bert = Bert()

bert.cfg.datamodule.datapath = 'path/to/file.csv'

bert.train()
```

<br><br>

That's all the code you need to start training. However, the csv file must be structured like this:

| id             | comment_text            | label_1 | label_2 |
|:--------------:|:-----------------------:|:-------:|:-------:|
| 12323434       | a sentence              | 0       | 1       |
| 12323435       | another one             | 1       | 0       |
| 12323436       | boy, is this repetitive | 0       | 1       |

<br><br>

---
## Changing Default Configurations

So you finished training and testing, but you'd like to tweak it more, you can do that using the configs. 

What configs can you change?

```yaml

model:
  model_id: "bert-base-cased"
  transformer_module: BertForSequenceClassification
  checkpoint_path: null

datamodule:
  tokenizer: BertTokenizerFast
  batch_size: 1      # 32 if using gpu
  max_token_len: 100 # 526 if using gpu
  num_workers: 0     # 2 if using gpu

trainer:
  max_epochs: 2
  progress_bar_refresh_rate: 1
  gpus: 0

checkpoint:
  name: checkpoint_${now:%Y-%m-%d}_${now:%H-%M-%S}

custom: false
```

<br><br>

Here's an example. To change the batch size in the configs:

```python

from mlfinance.nlp import Bert

bert = Bert()

bert.cfg.datamodule.batch_size = # insert batch size here

bert.train()
```

<br><br>

\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~ Summary of Configs \~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~

**model**
- **model_id**: transformer model id*
- **transformer_module**: transformer model*
- **checkpoint_path**: path/to/model.ckpt; does not accept .onnx files, yet

**datamodule**
- **tokenizer**: transformer tokenizer*
- **batch_size**: number of texts loaded into a batch 
- **max_token_len**: maximum number of words in a text
- **num_workers**: number of concurrent threads that load data

**trainer**
- **max_epochs**: maximum number of epochs before training finishes
- **progress_bar_refresh_rate**: how often the progress bar refreshes on command line 
- **gpus**: number of gpus, 0 if using cpu (yes, you can train on cpu)

\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~

<br><br>

\*The transformer model id, name, and tokenizer can be found by searching the docs [here](https://huggingface.co/models). 

The defaults are bert-base-cased, BertForSequenceClassification, and BertTokenizerFast, respectively.

Below is a small table of valid configurations. The transformers library is constantly growing, so don't stop here:


Model | model_id | name | tokenizer
--- | --- | --- | --- 
BERT |	bert-base-uncased, bert-large-uncased, bert-base-multilingual-uncased | BertForSequenceClassification | BertTokenizerFast
DistilBERT |	distilbert-base-uncased, and others | DistilBertForSequenceClassification | DistilBertTokenizerFast
RoBERTa |	roberta-base, roberta-large, roberta-large-mnli | RobertaForSequenceClassification | RobertaTokenizerFast
XLNet |	xlnet-base-cased, xlnet-large-cased | XLMForSequenceClassification | XLMTokenizer

<br><br>

---
## Testing Trained Model

You've finished training. Let's test it on new data that we've never seen before.

```python
from mlfinance.nlp import Bert
import torch.nn as nn

bert = Bert()

# load checkpoint (no ONNX support yet)
bert.cfg.model.checkpoint_path = 'path/to/model.ckpt'
# load csv file
bert.cfg.datamodule.datapath = 'path/to/new/data.csv'
bert.initialize() # always do this before testing on your own data

dataloader = bert.datamodule.test_dataloader()

for batch in iter(dataloader):
    loss, predictions = bert.model(model)
    print(nn.Sigmoid(predictions))
```
Any suggestions to make this process more straightforward are welcome.

<br><br>

---

## Command Line Workflow

All the configs can also be changed using the command line, without writing any python code! First, navigate to ml_finance/mlfinance/nlp. Then you can run this to change the batch size and checkpoint_path:

> python train.py datamodule.batch_size=4 model.checkpoint_path=path/to/model.ckpt

Other configs can be changed in a similar manner. You can essentially do all your training entirely from the command line.

<br><br>

---

## Advanced Features

Below are small snippits of functionality that are still in the experimental phase, but already work.

**Add Pytorch Lightning Callbacks**

```python
# More callbacks can be found here:
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html

from mlfinance.nlp import DistilBert
from pytorch_lightning.callbacks import GPUStatsMonitor

bert = DistilBert()
# This will monitor GPU usage
bert.callbacks.append(GPUStatsMonitor())

bert.train()

# View GPU usage on tensorboard
# tensorboard --logdir lightning_logs
```

**Accumulating Grad Batches for Larger Text Classification**

```python
# Currently, the maximum number of words
# it can train on is 512 words per batch
# But, with accumulate_grad_batches=2, it will
# train on essentially 1024 words per batch as
# long as you split up the training data correctly
# this is good when you want to train on larger 
# pieces of text

from mlfinance.nlp import DistilBert

# use Hydra overrides to ADD argument to the config
# this also works with bert
distilbert = DistilBert(overrides=[
  "+trainer.accumulate_grad_batches=2",                                
])

distilbert.train() # this won't be able to save to ONNX
# but the checkpoint is saved in lightning_logs!
# We may have to do convert it to ONNX manually in the future

# For more information on Hydra overrides:
# https://hydra.cc/docs/advanced/override_grammar/basic
# For more information on the pytorch-lightning trainer:
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
```


**Module Swapping**

```python
# Let's pretend for a moment that you would like to swap out the model module
# for one of your own, but would like to keep the training loop

from mlfinance.nlp import Bert

bert = Bert()

bert.initialize() # this modules load like normal from configs

bert.model = CustomModelClass() # insert class here

bert.train() # train like normal
```


**Model-Pruning with Bert**

```python
# This needs quite a bit of GPU/RAM, be careful

from mlfinance.nlp import Bert

bert = Bert()

bert.model_pruning = True

bert.train()
```


**Multi-run experiments using Hydra**

```bash
python mlfinance/nlp/train.py -m +trainer.max_epochs=10,20
```