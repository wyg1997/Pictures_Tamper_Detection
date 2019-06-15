import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog

from PIL import Image, ImageTk
from scipy.misc import imread
import cv2

import os.path as osp
import os
from pprint import pprint

import numpy as np
import torch
from torch.autograd import Variable
from model.faster_rcnn.vgg16 import vgg16
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.roi_layers import nms
from model.utils.blob import im_list_to_blob
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.utils.net_utils import vis_detections

checked_file = 0

# 点击"浏览"按钮，浏览打开图像
def get_path():
    # 打开图片
    path = filedialog.askopenfilename(
                            title='选择图片',
                            filetypes=(('jpg图片', '*.jpg'), ('png图片', '*.png'), ('tif图片', '*.tif'))
                        )
    if path is not None:
        # 更新目录和文件列表，并指向当前图片名
        dir, name = osp.split(path)
        img_path.set(dir+'/')
        file_list.delete(0, tk.END)
        for _, _, files in os.walk(dir):
            for index, file in enumerate(files):
                file_list.insert(index, file)
                if osp.basename(file) == name:
                    checked_file = index
            break
        file_list.selection_set(checked_file)
        # 检测图像
        prosess(path)

# 双击文件列表，打开文件
def open_file(event):
    name = file_list.get(file_list.curselection())
    prosess(osp.join(text_path.get(), name))

# 处理读入的图片
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

# 调用网络模型，检测图像并输出
# ToDo: 检测图像，并绘制、输出结果
def prosess(img_path):
    im_in = np.array(imread(img_path))
    # to RGB
    im = im_in[:,:,::-1]
    # get input inform
    blobs, im_scales = _get_image_blob(im)
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    # 准备input
    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.resize_(1, 1, 5).zero_()
    num_boxes.resize_(1).zero_()

    # detection
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = model(im_data, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    box_deltas = bbox_pred.data
    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
    box_deltas = box_deltas.view(1, -1, 4 * 2)
    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    inds = torch.nonzero(scores[:,1]>0.05).view(-1)

    # show image
    im2show = np.copy(im)

    # if there is det
    if inds.numel() > 0:
        cls_scores = scores[:, 1][inds]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes = pred_boxes[inds][:, 4:8]
        
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets = cls_dets[order]
        # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]

        # 调试可视化
        im2show = vis_detections(im2show, 'tampered', cls_dets.cpu().numpy(), 0.5)

        # 输出结果
        # cv2.imshow('test', im2show)
        # cv2.waitKey(0)
    else:
        print('No bbox!')

    # 暂存结果
    cv2.imwrite('temp.jpg', im2show)

    # to PIL
    img = Image.open('temp.jpg')
    img = img.resize((640, 480))

    img = ImageTk.PhotoImage(img)
    label_img.config(image=img)
    label_img.image = img
    pass

# 加载网络模型
def load_model(weights_path=None, net='vgg16'):
    cfg_file = './cfgs/vgg16.yml'
    cfg_from_file(cfg_file)
    cfg_from_list(['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]'])

    # output config
    print('model cfg:')
    pprint(cfg)

    model = vgg16(('background', 'tampered'), pretrained=False, class_agnostic=False)
    model.create_architecture()
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('Load model successfully!')
    model = model.cuda()
    model.eval()
    return model

# some var
im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)
# to cuda
im_data = im_data.cuda()
im_info = im_info.cuda()
num_boxes = num_boxes.cuda()
gt_boxes = gt_boxes.cuda()
# make variable
im_data = Variable(im_data, volatile=True)
im_info = Variable(im_info, volatile=True)
num_boxes = Variable(num_boxes, volatile=True)
gt_boxes = Variable(gt_boxes, volatile=True)

# load model
model = load_model('./models/vgg16/pascal_voc/faster_rcnn_1_20_1153.pth')

# 主界面绘制
window = tk.Tk()
window.title('Image Manipulation Detection')
window.geometry('1100x600')

# title
label_title = tk.Label(window, text='图像篡改检测', font=('Arial', 24))
label_title.pack(side='top', ipady=15)

# 画检测图像
# Test
temp_img_url = 'E:/GraduationDesign/Pictures_Tamper_Detection/data/casia/CASIA1/Sp/Sp_D_CNN_A_art0024_ani0032_0268.jpg'

img = Image.open(temp_img_url)
# Resize，使图片放大
# ToDo: 等比例放大
img = img.resize((640, 480))
img = ImageTk.PhotoImage(img)
label_img = tk.Label(window, image=img, compound=tk.CENTER)
label_img.pack(side='left')

# 文件列表（因为之前想法是画文件树，变量名叫tree）
file_tree = tk.Frame(window)
browser_frame = tk.Frame(file_tree)
file_list_frame = tk.Frame(file_tree)
# 提示文件，按钮和目录名
label_tip = tk.Label(browser_frame, text='请选择文件:')
label_tip.grid(row=0, column=0, padx=1, pady=0, ipadx=5, ipady=5)

img_path = tk.StringVar()
text_path = tk.Entry(browser_frame, textvariable=img_path, width=35, state=tk.DISABLED)
text_path.grid(row=0, column=1, padx=1, pady=0, ipadx=5, ipady=5)

button_browser = tk.Button(
                    browser_frame,
                    text='浏览',
                    command=get_path
                )
button_browser.grid(row=0, column=2, padx=1, pady=0, ipadx=5, ipady=5)

# 带滚动条的文件列表
scrollbar=tk.Scrollbar(file_list_frame)
scrollbar.pack(side=tk.RIGHT,fill=tk.Y)
file_list = tk.Listbox(file_list_frame, width=50, height=20, yscrollcommand=scrollbar.set, selectmode=tk.BROWSE)
# 添加双击事件
file_list.bind('<Double-Button-1>', open_file)

scrollbar.config(command=file_list.yview)
file_list.pack()

browser_frame.pack(side='top')
file_list_frame.pack(side='bottom')

file_tree.pack(side='right', ipadx=10, ipady=10)
# 文件列表绘制完成

window.mainloop()