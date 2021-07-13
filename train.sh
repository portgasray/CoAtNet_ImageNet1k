export CUDA_VISIBLE_DEVICES=0,1,2
nohup python -u coatnet.py  \
	--epochs 60  \
	-b 48  \
	--lr 0.001  \
	--resume './model/'  \
	--world-size 1  \
	--rank 0  \
	--dist-url 'tcp://127.0.0.1:1345'  \
        --dist-backend 'nccl'  \
	--multiprocessing-distributed  \
	data/ILSVRC2012/images > log/log_2.log 2>&1 &
