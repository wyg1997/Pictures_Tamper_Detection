import os

img_path = 'data/DIY_dataset/VOC2012/JPEGImages/'

save_dir = 'data/DIY_dataset/VOC2012/ImageSets/Main/'

for root, dir, files in os.walk(img_path):
    num_train = int(len(files) * 0.8)
    # num_test = len(files) - num_train

    with open(os.path.join(save_dir, 'trainval.txt'), 'w+') as f:
        for i in range(num_train):
            f.write(files[i][:-4] + '\n')

    with open(os.path.join(save_dir, 'test.txt'), 'w+') as f:
        for i in range(num_train, len(files)):
            f.write(files[i][:-4] + '\n')

