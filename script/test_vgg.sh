#!/usr/bin/env bash
python test_net.py \
--net vgg16 \
--dataset pascal_voc \
--checksession 1 \
--checkepoch 20 \
--checkpoint 1153 \
--vis \
--cuda