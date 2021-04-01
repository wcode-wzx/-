import  torch
from torch.autograd import Variable
from mnist_train import Net #引入模型
import  torch.nn.functional as F#激活函数
import torch.optim as optim
import cv2 as cv
import numpy as np 
from PIL import Image

'''
- 只加载了权重参数进行预测
- 预测时加载自己的图片时，注意将图片转化成训练时的格式
'''
model = Net()#实例化模型

#调用模型参数
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load('w_m/parameter.pkl'))
model.eval()

#把计算迁移到GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#定义一个损失函数，来计算我们模型输出的值和标准值的差距
criterion = torch.nn.CrossEntropyLoss()
#定义一个优化器，训练模型咋训练的，就靠这个，他会反向的更改相应层的权重
optimizer = optim.SGD(model.parameters(),lr=0.1,momentum=0.5)#lr为学习率

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
    outputs = model(img)
    print(outputs)
    #test()
    _, pred = torch.max(outputs, 1)
    print('预测为:数字{}。'.format(pred))

    