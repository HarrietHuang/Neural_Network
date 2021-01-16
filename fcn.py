#ref: https://github.com/bat67/pytorch-FCN-easiest-demo
from google.colab import files as gfile
from google.colab import drive
drive.mount('/content/gdrive')
import os
import torch as t
import torch.utils.data as data
from PIL import Image
from torchvision import  transforms as T
import os
import numpy as np
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from IPython import display
#
from torchvision import  transforms as T
import torch.utils.data as data
import os
#dataset, dataloader
# 資料夾
imfile_array = [
    'high         ',
    'medium       ',
    'low          ',
    'high_noblur    ',
    'medium_noblur  ',
    'low_noblur     ']

#google drive 上的路徑
root = '/content/gdrive/My Drive/Colab Notebooks' 
transforms = T.Compose([T.Resize((224,224)),T.ToTensor()])

#T.compose 可以把很多東西都合起來
#https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomCrop

Test_propotion = 2 # n out of len(imfile_array)=12 are set to test

class datasets(data.Dataset):
    def __init__(self, root, propotion,train = True, transforms = None) :
        imgs = []
        if train == True:
          imfile = imfile_array[:-propotion]
        else:
          imfile = imfile_array[-propotion:]

      
        for dirss in imfile: #os.listdir(root)[:-propotion]:
            if os.path.isdir(os.path.join(root,dirss.strip())):
                for files in os.listdir(os.path.join(root,dirss.strip())):
                    imgs.append(files)
        imgs = [lab for lab in imgs if lab.endswith('_noblur.png')]  
        #imgs = [lab for lab in imgs if '_noblur' in lab]  
        #clear
 
        
        self.imgs = imgs 
        self.transforms = transforms
       # print(imgs)        
        
    def __getitem__(self,index):
        imfile = self.imgs[index].split('(')[0][:-1]
        blur_img_path = os.path.join(root,imfile,self.imgs[index])
        noblur_img_path = os.path.join(root,imfile+"_noblur",self.imgs[index])
        
        #if img_path.endswith('_noblur.png'):
        label = Image.open(noblur_img_path)
        blur = Image.open(blur_img_path.replace('_noblur.png','.bmp'))

        if self.transforms:
            blur = self.transforms(blur)
            label = self.transforms(label)
        return blur, label
    
    def __len__(self):
        return len(self.imgs)


    
train_dataset = datasets(root, Test_propotion,True ,transforms)
test_dataset = datasets(root, Test_propotion,False ,transforms)

print(len(train_dataset))
print(len(test_dataset))
#train_dataset, test_dataset = data.random_split(dataset, [len(dataset)-3,3])

train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=0)
test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True,num_workers=0)

# for  k, l in enumerate(train_dataloader):
#    print(k, l[0].size())
#    print(l[1].size())

ranges = {
   # 'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
   # 'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
   # 'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

cfg = {
   # 'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
   # 'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
   # 'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def addpad(big, small):
  """add pads to small as size of big"""
  diffY = big.size()[2] - small.size()[2]
  diffX = big.size()[3] - small.size()[3]
  small = F.pad(small, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

  return small  



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=False)] 
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
#    print('VGGnet=',nn.Sequential(*layers))
    return nn.Sequential(*layers)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]
#        print(self._modules.keys())

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # delete redundant fully-connected layer params, can save memory
        # remove(classifier)
        if remove_fc:  
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx, (begin, end) in enumerate(self.ranges):
        #self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)

          for layer in range(begin, end):
            x = self.features[layer](x)
            if layer == 0:
              output["x0"] = x
            output["x%d"%(idx+1)] = x
#            print('layer',layer,idx+1,x.size())
        #print(output)
        return output

class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
#        print(self._modules.keys())
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1) 
        # classifier is 1x1 conv, to reduce channels from 32 to n_class

    def forward(self, x):
        output = self.pretrained_net(x) #output is a dic
        x5 = output['x5']  
#        print('x5.size',x5.size())
        x4 = output['x4']  
#        print('x4.size',x4.size())
        x3 = output['x3']
#        print('x3.size',x3.size())
        x2 = output['x2']  
#        print('x2.size',x2.size())
        x1 = output['x1']  
#        print('x1.size',x1.size())
        x0 = output['x0'] 
#        print('x0.size',x0.size())
        
        score = self.bn1(self.relu(self.deconv1(x5)))
        score = addpad(x4,score)
        score = score + x4  

        
        score = self.bn2(self.relu(self.deconv2(score)))
        score = addpad(x3,score)
        score = score + x3    

        
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = addpad(x2,score)
        score = score + x2


        score = self.bn4(self.relu(self.deconv4(score)))  
        score = addpad(x1,score)
        score = score + x1   


        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    
        
#        print('score',score.size())
        
        return addpad(x0,score)
        #Train 
vgg_model = VGGNet(requires_grad=True, show_params=False)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=3)
def trainandtest(fcn_model, epo_num=50, show_vgg_params=False,evaluate=False,load_state_path=None,downloadimg=False):

  device = t.device('cuda' if t.cuda.is_available() else 'cpu')
  
  
  if evaluate == True: #load_state_dict(torch.load(model_path))
    pth = t.load(load_state_path)
    fcn_model.load_state_dict(pth) 
  
  fcn_model = fcn_model.to(device)
  criterion = nn.MSELoss().to(device)
  #optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.75,weight_decay=0.01)
  #optimizer = optim.Adam(fcn_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
  optimizer = optim.RMSprop(fcn_model.parameters(), lr=0.01, alpha=0.9)



  # start timing
  prev_time = datetime.now()
  lastepochloss = 0
  iter_loss = 0
  if evaluate == True:
    epo_num = 1
  for epo in range(epo_num):
    train_loss = 0
    fcn_model.train()
    tq = tqdm(iter(train_dataloader), leave=False, total=len(train_dataloader))
    for index, (blur, groundtruth) in enumerate(tq):
      if evaluate == True:
        break
      tq.set_description('Epoch %i'%(epo))
      blur = blur.to(device)
      groundtruth = groundtruth.to(device)

      optimizer.zero_grad()
      output = fcn_model(blur)
#        output = t.sigmoid(output) 
      loss = criterion(output, groundtruth)
      loss.backward()
      iter_loss = loss.item()
      train_loss += iter_loss
      optimizer.step()

      

      if np.mod(index, 15) == 0:
          tq.set_postfix(Last_epoch_Loss = lastepochloss ,Train_Loss = iter_loss)
#          tqdm.write('\n epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))

    # eval
    #   

    test_loss = 0
    fcn_model.eval()

    with t.no_grad():
      if epo == epo_num-1: downloadimg = True
#      tqtest = tqdm(iter(test_dataloader), leave=False, total=len(test_dataloader))
      for index, (blur, groundtruth) in enumerate(test_dataloader):

        blur = blur.to(device)
        groundtruth = groundtruth.to(device)

        optimizer.zero_grad()
        output = fcn_model(blur)
#        output = t.sigmoid(output) 
        loss = criterion(output, groundtruth)
        iter_loss = loss.item()
        test_loss += iter_loss
        
        if downloadimg == True: #Combine groundtruth, output,and input and then download
            
          for bsize in range(test_dataloader.batch_size): 
            
            image_list = [T.ToPILImage()(groundtruth.cpu()[bsize]), T.ToPILImage()(output.cpu()[bsize]),T.ToPILImage()(blur.cpu()[bsize])]

            width, height = T.ToPILImage()(groundtruth.cpu()[bsize]).size

            new_im = Image.new('RGB', (width*len(image_list), height))

            x_offset = 0
            for im in image_list:
              new_im.paste(im, (x_offset,0))
              x_offset += im.size[0]

            new_im.save('test.png','png')
            #gfile.download('test.png')
            display.display(new_im)
            




    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    prev_time = cur_time
    lastepochloss = test_loss/len(test_dataloader)
       
    print('\n epoch train loss = %f, epoch test loss = %f, %s'%(train_loss/len(train_dataloader), lastepochloss, time_str))
    
    
    

#    if np.mod(epo, 5) == 0:
#        t.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
#        print('saveing checkpoints/fcn_model_{}.pt'.format(epo))

#train(epo_num=50, show_vgg_params=False)

#train : evaluate=False downloading = false
#adjust parameter

trainandtest(fcn_model,epo_num=400, show_vgg_params=False,evaluate=False,load_state_path='/content/gdrive/My Drive/Colab Notebooks/deblur',downloadimg=False)
t.save(fcn_model.state_dict(), '/content/gdrive/My Drive/Colab Notebooks/deblur')
