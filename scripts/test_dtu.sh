#!/usr/bin/env bash
DTU_TESTPATH="/hdd1/lhs/dev/dataset/dtu_test_all"
DTU_TESTLIST="lists/dtu/test.txt"

exp=$1

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_CKPT_FILE=$DTU_LOG_DIR"/finalmodel.ckpt"
DTU_OUT_DIR="./outputs/dtu/"$exp


python test_mvs4.py --dataset=general_eval4 --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
             --inverse_depth --group_cor | tee -a $DTU_LOG_DIR/log_test.txt
