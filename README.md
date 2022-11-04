# Turning Fixed to Adaptive: Integrating Post-Evaluation into Simultaneous Machine Translation

Source code for our EMNLP 2022 paper ["Turning Fixed to Adaptive: Integrating Post-Evaluation into Simultaneous Machine Translation
"](https://arxiv.org/abs/2210.11900) 

Our method is implemented based on the open-source toolkit [Fairseq](https://github.com/facebookresearch/fairseq) .



## Requirements and Installation

* Python version = 3.8

* PyTorch version = 1.10

* Install fairseq:

```
git clone https://github.com/ictnlp/PED-SiMT.git
cd PED-SiMT-main
pip install --editable ./
```


## Quick Start

### Data Pre-processing

We use the data of IWSLT15 English-Vietnamese (download [here](https://nlp.stanford.edu/projects/nmt/)), IWSLT14 English-German (download [here](https://wit3.fbk.eu/2014-01)) and WMT15 German-English (download [here](www.statmt.org/wmt15/)).

For WMT15 German-English, we tokenize the corpus via [mosesdecoder/scripts/tokenizer/normalize-punctuation.perl](https://github.com/moses-smt/mosesdecoder) and apply BPE with 32K merge operations via [subword_nmt/apply_bpe.py](https://github.com/rsennrich/subword-nmt). Follow [preprocess scripts](https://github.com/Vily1998/wmt16-scripts) to perform tokenization and BPE.

Then, we process the data into the fairseq format, adding ```--joined-dictionary``` for WMT15 German-English:

```
src=SOURCE_LANGUAGE
tgt=TARGET_LANGUAGE
train_data=PATH_TO_TRAIN_DATA
vaild_data=PATH_TO_VALID_DATA
test_data=PATH_TO_TEST_DATA
data=PATH_TO_PROCESSED_DATA

# add --joined-dictionary for WMT15 German-English
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
```

### Training

Train our model on WMT15 German-English with the following command:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

data=PATH_TO_TRAIN_DATA
modelfile=PATH_TO_SAVE_CHECKPOINTS

python train.py --ddp-backend=no_c10d ${data} --arch transformer \
 --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 \
 --left-pad-source False \
 --save-dir ${modelfile} \
 --max-tokens 2048 --update-freq 4 \
 --save-interval-updates 1000 \
 --keep-interval-updates 500 \
 --log-interval 10
```

### Inference

Evaluate the model with the following command:

```
export CUDA_VISIBLE_DEVICES=0
data=PATH_TO_TEST_DATA
modelfile=PATH_TO_SAVE_CHECKPOINTS
ref_dir=PATH_TO_REFERENCE
k=NUM_PREREAD_TOKENS
threshold=PREDEFINED_PARAMETER

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt 

# generate translation
python generate.py ${data} --path $modelfile/average-model.pt --batch-size 1 --beam 1 --left-pad-source False --remove-bpe --gen-subset test --sim-decoding --latency &{k} --threshold ${threshold} > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${ref_dir} < pred.translation
```

## Citation
```
@inproceedings{PED-SiMT,
    title = "Turning Fixed to Adaptive: Integrating Post-Evaluation into Simultaneous Machine Translation",
    author = "Guo, Shoutao and Zhang, Shaolei and Feng, Yang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Online and Abu Dhabi",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2210.11900",
}
```
