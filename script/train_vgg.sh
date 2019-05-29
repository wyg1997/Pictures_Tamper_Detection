CUDA_VISIBLE_DEVICES=2,3 python trainval_net.py \
	--dataset pascal_voc \
	--net vgg16 \
	--bs 32 \
	--nw 2 \
	--lr 0.001 \
	--lr_decay_step 5 \
	--cuda
