#!/usr/bin/env bash
python test_net.py \
--net vgg16 \
--dataset pascal_voc \
--checksession 1 \
--checkepoch 3 \
--checkpoint 4615 \
--vis \
--cuda