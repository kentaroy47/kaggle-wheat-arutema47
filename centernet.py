#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision


# In[2]:


input_size = 512
IN_SCALE = 1024//input_size
MODEL_SCALE = 4
batch_size = 4
model_name = "resnet18"


# ## Prepare labels

# In[3]:


DIR_INPUT = 'data'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
train_df.shape


# In[4]:


train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)


# In[5]:


# Split train-test
from sklearn.model_selection import train_test_split
# Split by unique image ids.
image_ids = train_df['image_id'].unique()
train_id, test_id = train_test_split(image_ids, test_size=0.2, random_state=777)


# In[6]:


test_id.shape


# In[7]:


train_id.shape


# ## convert boxes to heatmap

# In[8]:


train_df[:5]


# In[9]:


# show image
img_id = train_id[0]
img = cv2.imread(os.path.join(DIR_INPUT,"train", img_id+".jpg"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)


# In[10]:


# get targets
target = train_df[train_df['image_id']==img_id]
print(target)
# convert targets to its center.
try:
    center = np.array([target["x"]+target["w"]//2, target["y"]+target["h"]//2]).T
except:
    center = np.array([int(target["x"]+target["w"]//2), int(target["y"]+target["h"]//2)]).T.reshape(1,2)
center


# In[11]:


# plot centers on image
plt.figure(figsize=(14,14))
plt.imshow(img)
for x in center:
    print(x)
    plt.scatter(x[0], x[1], color='red', s=100)


# In[12]:


# Make heatmaps using the utility functions from the centernet repo
def draw_msra_gaussian(heatmap, center, sigma=2):
  tmp_size = sigma * 6
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap
def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
  dim = value.shape[0]
  reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
  if is_offset and dim == 2:
    delta = np.arange(diameter*2+1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


# In[13]:


# get targets
target = train_df[train_df['image_id']==img_id]
# convert targets to its center
try:
    center = np.array([target["x"]+target["w"]//2, target["y"]+target["h"]//2, 
                   target["w"], target["h"]
                  ]).T
except:
    center = np.array([int(target["x"]+target["w"]//2), int(target["y"]+target["h"]//2), 
                   target["w"], target["h"]
                  ]).T.reshape(1,4)

# make output heatmap for single class
hm = np.zeros([input_size//MODEL_SCALE, input_size//MODEL_SCALE])
# make a center point
# try gaussian points.
for c in center:
    hm = draw_msra_gaussian(hm, [int(c[0])//MODEL_SCALE, int(c[1])//MODEL_SCALE], 
                            sigma=np.clip(c[2]*c[3]//2000, 2, 4))

# plot and confirm if its working
plt.imshow(hm)
plt.show()
plt.imshow(cv2.resize(img, (128,128))+
           (np.array([hm,hm,hm]).transpose([1,2,0])*255).astype("int8"))
plt.show()


# In[14]:


# make regr heatmap 
regr = np.zeros([2, input_size//MODEL_SCALE, input_size//MODEL_SCALE])

# convert targets to its center.
regrs = center[:, 2:]/input_size

# plot regr values to mask
for r, c in zip(regrs, center):
    x,y = int(c[0])//MODEL_SCALE, int(c[1])//MODEL_SCALE
    for i in range(-2, 3):
        for j in range(-2, 3):
            try:
                regr[:, int(c[0])//MODEL_SCALE+i, int(c[1])//MODEL_SCALE+j] = r
            except:
                pass
regr[0] = regr[0].T; regr[1] = regr[1].T;
print("show regr")
plt.imshow(regr[0])
plt.show()
plt.imshow(regr[1])
plt.show()


# In[15]:


def make_hm_regr(target):
    # make output heatmap for single class
    hm = np.zeros([input_size//MODEL_SCALE, input_size//MODEL_SCALE])
    # make regr heatmap 
    regr = np.zeros([2, input_size//MODEL_SCALE, input_size//MODEL_SCALE])
    
    if len(target) == 0:
        return hm, regr
    
    try:
        center = np.array([target["x"]+target["w"]//2, target["y"]+target["h"]//2, 
                       target["w"], target["h"]
                      ]).T
    except:
        center = np.array([int(target["x"]+target["w"]//2), int(target["y"]+target["h"]//2), 
                       int(target["w"]), int(target["h"])
                      ]).T.reshape(1,4)
    
    # make a center point
    # try gaussian points.
    for c in center:
        hm = draw_msra_gaussian(hm, [int(c[0])//MODEL_SCALE//IN_SCALE, int(c[1])//MODEL_SCALE//IN_SCALE], 
                                sigma=np.clip(c[2]*c[3]//2000, 2, 4))    

    # convert targets to its center.
    regrs = center[:, 2:]/input_size/IN_SCALE

    # plot regr values to mask
    for r, c in zip(regrs, center):
        for i in range(-2, 3):
            for j in range(-2, 3):
                try:
                    regr[:, int(c[0])//MODEL_SCALE//IN_SCALE+i, 
                         int(c[1])//MODEL_SCALE//IN_SCALE+j] = r
                except:
                    pass
    regr[0] = regr[0].T; regr[1] = regr[1].T;
    return hm, regr


# In[16]:


def pred2box(hm, regr, thresh=0.99):
    # make binding box from heatmaps
    # thresh: threshold for logits.
        
    # get center
    pred = hm > thresh
    pred_center = np.where(hm>thresh)
    # get regressions
    pred_r = regr[:,pred].T

    # wrap as boxes
    # [centerx, centery, width, height]
    # size as original image.
    boxes = []
    for i, b in enumerate(pred_r):
        boxes.append([pred_center[1][i]*MODEL_SCALE, pred_center[0][i]*MODEL_SCALE, int(b[0]*input_size), int(b[1]*input_size)])
    return np.asarray(boxes)


# ## compile data processing

# In[17]:


train_df[train_df['image_id']==train_id[2476]]


# In[18]:


# show image
#img_id = train_df["image_id"][]
img_id = train_id[2476]
img = cv2.imread(os.path.join(DIR_INPUT,"train", img_id+".jpg"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (input_size, input_size))
sample = img

# get labels
target = train_df[train_df['image_id']==img_id]

# convert target to heatmaps
hm, regr = make_hm_regr(target)

# get boxes
boxes = pred2box(hm, regr)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
for box in boxes:
    cv2.rectangle(sample,
                  (box[0]-box[2]//2, box[1]+box[3]//2),
                  (box[0]+box[2]//2, box[1]-box[3]//2),
                  (220, 0, 0), 3)
plt.imshow(sample)
plt.show()


# ## make dataset

# In[19]:


from torchvision import transforms

class Normalize(object):
    def __init__(self):
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.norm = transforms.Normalize(self.mean, self.std)
    def __call__(self, image):
        image = image.astype(np.float32)/255
        axis = (0,1)
        image -= self.mean
        image /= self.std
        return image

class WheatDataset(torch.utils.data.Dataset):
    def __init__(self, img_id, labels, transform=None):
        self.img_id = img_id
        self.labels = labels
        if transform:
            self.transform = transform
        self.normalize = Normalize()
        
    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(DIR_INPUT,"train", self.img_id[idx]+".jpg"))
        img = cv2.resize(img, (input_size, input_size))
        img = self.normalize(img)
        img = img.transpose([2,0,1])
        target = self.labels[self.labels['image_id']==self.img_id[idx]]
        hm, regr = make_hm_regr(target)
        return img, hm, regr


# In[20]:


target = train_df[train_df['image_id']==train_id[2476]]
target["h"]


# In[21]:


traindataset = WheatDataset(train_id, train_df)
valdataset = WheatDataset(test_id, train_df)
train_df
img, hm, regr = traindataset[0]
plt.imshow(img.transpose([1,2,0]))
plt.show()
img.std()
plt.imshow(hm)


# In[22]:


train_loader = torch.utils.data.DataLoader(traindataset,batch_size=batch_size,shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(valdataset,batch_size=batch_size,shuffle=True, num_workers=0)


# ## Import models

# In[23]:


from model import centernet
model = centernet()
model(torch.rand(1,3,512,512))[0].size()


# In[24]:


# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# In[25]:


from loss import centerloss
criterion = centerloss


# # Train models

# In[26]:


def train(epoch):
    model.train()
    print('epochs {}/{} '.format(epoch+1,epochs))
    running_loss = 0.0
    running_mask = 0.0
    running_regr = 0.0
    t = tqdm(train_loader)
    rd = np.random.rand()
    
    for idx, (img, hm, regr) in enumerate(t):       
        # send to gpu
        img = img.to(device)
        hm_gt = hm.to(device)
        regr_gt = regr.to(device)
        # set opt
        optimizer.zero_grad()
        
        # run model
        hm, regr = model(img)
        preds = torch.cat((hm, regr), 1)
            
        loss, mask_loss, regr_loss = criterion(preds, hm_gt, regr_gt)
        # misc
        running_loss += loss
        running_mask += mask_loss
        running_regr += regr_loss
        
        loss.backward()
        optimizer.step()
        
        t.set_description(f't (l={running_loss/(idx+1):.3f})(m={running_mask/(idx+1):.4f})(r={running_regr/(idx+1):.4f})')
        
    #scheduler.step()
    print('train loss : {:.4f}'.format(running_loss/len(train_loader)))
    print('maskloss : {:.4f}'.format(running_mask/(len(train_loader))))
    print('regrloss : {:.4f}'.format(running_regr/(len(train_loader))))
    
    # save logs
    log_epoch = {'epoch': epoch+1, 'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                    'loss': running_loss/len(train_loader), "mask": running_mask/(len(train_loader)), 
                 "regr": running_regr/(len(train_loader))}
    logs.append(log_epoch)
    df = pd.DataFrame(logs)
    os.makedirs("log", exist_ok=True)
    df.to_csv("log/log_output_train_{}.csv".format(model_name))


# In[ ]:


import gc
os.makedirs("models", exist_ok=True)
epochs=100
logs = []

for epoch in range(epochs):
    train(epoch)
    if epoch%5==0:
        torch.save(model.state_dict(), './models/{}_{}epochs_saved_weights.pth'.format(model_name, epoch))
        
    # GC
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:


img, _, _ = traindataset[0]
img = torch.from_numpy(img)
with torch.no_grad():
    hm, regr = model(img.to(device).float().unsqueeze(0))


# In[ ]:


plt.imshow(hm.cpu().numpy().squeeze(0).squeeze(0)>0.8)
hm = hm.cpu().numpy().squeeze(0).squeeze(0)
regr = regr.cpu().numpy().squeeze(0)


# In[ ]:


img = (img.numpy().transpose([1,2,0])*255).astype("int")
plt.imshow(img)


# In[ ]:


# show image
img_id = train_id[0]
img = cv2.imread(os.path.join(DIR_INPUT,"train", img_id+".jpg"))
img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (input_size, input_size))

# get boxes
boxes = pred2box(hm, regr, thresh=0.75)
sample = img

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
for box in boxes:
    cv2.rectangle(sample,
                  (int(box[0]-box[2]//2), int(box[1]+box[3]//2)),
                  (int(box[0]+box[2]//2), int(box[1]-box[3]//2)),
                  (220, 0, 0), 3)
plt.imshow(sample)
plt.show()


# In[ ]:




