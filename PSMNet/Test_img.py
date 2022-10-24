from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image
from pathlib import Path
from skimage import io
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
"训练集"
# parser.add_argument('--KITTI', default='2015',
#                     help='KITTI version')
#parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
#                    help='select model')
"模型数据读取"
parser.add_argument('--loadmodel', default='D:\\PSMNet\\PSMNet\\pretrained_sceneflow_new.tar',
                    help='loading model')
"左右视图"
parser.add_argument('--leftimg', default='D:\\match\\stereo\\MiddEval3\\trainingQ\\PianoL\\im0.png',
                    help='load model')
parser.add_argument('--rightimg', default='D:\\match\\stereo\\MiddEval3\\trainingQ\\PianoL\\im1.png',
                    help='load model')


parser.add_argument('--model', default='stackhourglass',
                    help='select model')
"最大视差值"
parser.add_argument('--maxdisp', type=int, default=256,
                    help='maxium disparity')
"是否使用cuda训练"
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
"种子点"
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
""
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()     

        with torch.no_grad():
            disp = model(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp

#展示.pfm文件方法
def Show_pfm(img,win_name='pfm_image'):
    img=cv2.imread(img,0)
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # img = cv2.threshold(img,255,255,cv2.THRESH_BINARY)
        print(img.shape)
        cv2.imshow(win_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)

def main():

        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}

        "torch的转换：array转tensor，normalize归一化"
        transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])
        "图片读取"
        imgL_o = cv2.imread(args.leftimg)
        imgL_o=cv2.cvtColor(imgL_o,cv2.COLOR_BGR2RGB)
        imgR_o = cv2.imread(args.rightimg)
        imgR_o = cv2.cvtColor(imgR_o, cv2.COLOR_BGR2RGB)
        "图片转换"
        imgL = transform(imgL_o)
        imgR = transform(imgR_o)

        # pad to width and hight to 16 times
        # 计算填充的数据
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0
        "填充"
        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        "开始时间"
        start_time = time.time()
        "开始计算视差图"
        pred_disp = test(imgL,imgR)
        "结束时间"
        end_time=time.time()
        "输出时间"
        print('time = %.2f' %(end_time - start_time))
        # file=open("./Result/PianoL/timePSMNet.txt","w")
        # file.write(end_time - start_time)
        # file.close()

        if top_pad !=0 and right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            img = pred_disp[:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            img = pred_disp[top_pad:,:]
        else:
            img = pred_disp
        "打印结果"
        # print('img',img)
        img = Image.fromarray(img)
        img = np.ascontiguousarray(img)

        cv2.imwrite('./Result/PianoL/disp0PSMNet.pfm',img)
        #展示pfm图片
        Show_pfm('./Result/PianoL/disp0PSMNet.pfm',"pfm_image")
        # cv2.imshow("picure",img)
        # cv2.waitKey(0)
        #视差图转换成深度图
        # src = 'D:\\xiaohua\\PSMNet-master\\Test_disparity3_1.pfm'
        # f = 2852.758
        # doffs = 125.36
        # baseline = 178.089
        # d=260
        # depth_map = create_depth_map(src,baseline,f,d,doffs)
        # print("depth_map",depth_map)
        # img = Image.fromarray(depth_map)
        # img.show()
        # img.save('D:/xiaohua/MiddEval3/trainingF/Adirondack/disp.png')
        # img = np.ascontiguousarray(img)
        # cv2.imwrite('D:/xiaohua/MiddEval3/trainingF/Adirondack/disp0PSMNet.pfm',img)


if __name__ == '__main__':
   main()







