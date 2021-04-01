import  torch
from torch.autograd import Variable
from mnist_train import Net #引入模型
import cv2 as cv
import numpy as np 
from PIL import Image

'''
引用pt格式保存模型和权重，仅需加载from mnist import Net即可
'''

def p_in(path):
    input_image = path
    im = Image.open(input_image).resize((28, 28))     #取图片数据
    im = im.convert('L')      #灰度图
    im_data = np.array(im)
    im_data = torch.from_numpy(im_data).float()
    im_data = im_data.view(1, 1, 28, 28)
    if torch.cuda.is_available():
        img = Variable(im_data).cuda()
    #img_cv_2 = np.transpose(tensor_cv.numpy(), (1, 2, 0))
    return img

if __name__=='__main__':
   
    img = p_in('1.png')
    #读取保存的模型和参数
    new_m = torch.load('w_m/model.pt')
    outputs = new_m(img)
    _, pred = torch.max(outputs, 1)
    print('预测为:数字{}。'.format(pred))