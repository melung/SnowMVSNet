#!/usr/bin/env bash
DTU_TRAINING="/hdd1/lhs/dev/dataset/mvs_training/dtu/"
DTU_TRAINLIST="lists/dtu/train.txt"
DTU_TESTLIST="lists/dtu/test.txt"

exp=$1

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
DTU_CKPT_FILE=$DTU_LOG_DIR"/finalmodel.ckpt"
DTU_OUT_DIR="./outputs/dtu/"$exp

python -m torch.distributed.launch --nproc_per_node=6 train_mvs4.py --logdir $DTU_LOG_DIR --dataset=dtu_yao4 --batch_size=2 --trainpath=$DTU_TRAINING --summary_freq 100 \
                --group_cor --rt --inverse_depth --attn_temp 2 --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST | tee -a $DTU_LOG_DIR/log.txt

