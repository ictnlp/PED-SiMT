train.sh

python /data/guoshoutao/PED-SiMT/fairseq/train.py /data/guoshoutao/wmt15_de_en_bpe32k \
--arch transformer \
--share-all-embeddings \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-07 \
--warmup-updates 4000 \
--lr 5e-4 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 2048 \
--update-freq 4 \
--no-progress-bar \
--log-format json \
--left-pad-source False \
--encoder-attention-heads 8  \
--decoder-attention-heads 8 \
--ddp-backend=no_c10d \
--clip-norm 0.0 \
--dropout 0.3

generate.sh

k = 5

a = 0.24

python /data/guoshoutao/PED-SiMT/fairseq/generate.py /data/guoshoutao/wmt15_de_en_bpe32k \
--path checkpoint_best.pt \
--batch-size 1 --beam 1 \
--remove-bpe --gen-subset test \
--left-pad-source False \
--left-pad-target False \
--latency &k --threshold $a
