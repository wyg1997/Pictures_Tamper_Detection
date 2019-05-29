import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog

from PIL import Image, ImageTk

import os.path as osp
import os

import torch
from model.faster_rcnn.vgg16 import vgg16

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

# 调用网络模型，检测图像并输出
# ToDo: 检测图像，并绘制、输出结果
def prosess(img_path):
    img = Image.open(img_path)
    img = img.resize((640, 480))
    img = ImageTk.PhotoImage(img)
    label_img.config(image=img)
    label_img.image = img
    pass

# 加载网络模型
def load_model(weights_path=None, net='vgg16'):
    model = vgg16(2, pretrained=False, class_agnostic=False)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    return model

# 主界面绘制
window = tk.Tk()
window.title('Image Manipulation Detection')
window.geometry('1100x600')

# title
label_title = tk.Label(window, text='图像篡改检测', font=('Arial', 24))
label_title.pack(side='top', ipady=15)

# 画检测图像
# Test
temp_img_url = 'D:/personal/casia/CASIA1/Sp/Sp_D_CNN_A_art0024_ani0032_0268.jpg'

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