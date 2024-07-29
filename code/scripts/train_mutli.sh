CUDA_VISIBLE_DEVICES=0,1 \
 python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=11277 code/src/train_multi.py \
        --cfg code/cfg/birds_multi.yml