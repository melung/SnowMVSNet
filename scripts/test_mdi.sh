#!/usr/bin/env bash
MDI_TESTPATH="/hdd1/lhs/dev/code/MVSTER/data"
MDI_TESTLIST="lists/mdi/test.txt"

exp=$1


DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_CKPT_FILE=$DTU_LOG_DIR"/finalmodel.ckpt"
MDI_OUT_DIR="./outputs/mdi/"$exp



python test_mvs4.py --dataset=general_eval4_mdi --batch_size=1 --testpath=$MDI_TESTPATH  --testlist=$MDI_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $MDI_OUT_DIR\
             --inverse_depth --group_cor --max_h 1920 --max_w 1920 | tee -a $DTU_LOG_DIR/log_test.txt
