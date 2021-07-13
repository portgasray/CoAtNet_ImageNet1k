export CUDA_VISIBLE_DEVICES=0,1,2
nohup python -u coatnet.py  \
	--epochs 60  \
	-b 48  \
	--lr 0.01  \
	--resume './model/'  \
	--world-size 1  \
	--rank 0  \
	--dist-url 'tcp://127.0.0.1:1345'  \
        --dist-backend 'nccl'  \
	--multiprocessing-distributed  \
	data/ILSVRC2012/images > /home/ns-lzhang/workspace/Experiments/CoAtNet_ImageNet1k/imagenet_coatnet.log 2>&1 &
