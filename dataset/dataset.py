import numpy as np
from torchvision import transforms
from sklearn.utils import shuffle
import random
from utils import pascal2coco, coco2pascal
import torch
import cv2

class WheatDataset(torch.utils.data.Dataset):
    def __init__(self, config, img_id, labels, transform=None, train=False):
        self.img_id = img_id
        self.labels = labels
        if transform:
            self.transform = transform
        else:
            self.transform = False
        self.train = train
        self.config = config

    def __getitem__(self, idx):
        image_id = self.img_id[idx]
        rd = random.random()
        if (not self.train or rd < 0.5) or not self.config.train["cutmix"]:
            image, boxes = self.load_image_and_boxes(idx)
        elif self.config.train["mixup"] and rd > 0.75:
            num = 0
            # filter poorly cut images.
            while num <10:
                image, boxes = self.load_mixup_image_and_boxes(idx)
                num = len(boxes)
        else:
            num = 0
            # filter poorly cut images.
            while num <10:
                image, boxes = self.load_cutmix_image_and_boxes(idx)      
                num = len(boxes)
        
        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        # pascal2coco
        boxes = pascal2coco(boxes)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        
        for i in range(100):
            sample = self.transform(**{
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            })
            if len(sample['bboxes']) > 0:
                image = sample['image']
                target["boxes"] = sample["bboxes"]
                break
                
        center = box2center(np.array(target["boxes"]))
        # Make HMs
        if not self.train or self.config.data["input_size"]==self.config.data["train_size"]:
            hm, regr, regr_wh = make_hm_regr(self.config, center, pass_center=True)
        else:
            hm, regr, regr_wh = make_hm_regr(self.config, center, pass_center=True, train=True)
            
        hm = torch.tensor(hm)
        regr = torch.tensor(np.concatenate((regr, regr_wh)))
                
        return image, hm, regr

    def __len__(self) -> int:
        return self.img_id.shape[0]
    
    def load_image_and_boxes(self, index):
        image_id = self.img_id[index]
        image = cv2.imread('{}/{}.jpg'.format(self.config.data["train_dir"], image_id), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255
        records = self.labels[self.labels['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes
    
    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.img_id.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes  
    
    def load_mixup_image_and_boxes(self, index):
        image, boxes = self.load_image_and_boxes(index)
        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.img_id.shape[0] - 1))
        return (image+r_image)/2, np.vstack((boxes, r_boxes)).astype(np.int32)

def target2box(target):
    try:
        box = np.array([target["x"], target["y"], 
                       target["w"], target["h"]
                      ]).T
    except:
        box = np.array([int(target["x"]), int(target["y"]), 
                       int(target["w"]), int(target["h"])
                      ]).T.reshape(1,4)
    return box

def box2center(box):
    box = np.array(box).reshape(-1, 4)
    center = np.array([box[:,0]+box[:,2]/2, box[:,1]+box[:,3]/2, box[:,2], box[:,3]]).T
    return center

def pred2box(config, hm, regr, regr_wh=None, thresh=0.99):
    # make binding box from heatmaps
    # thresh: threshold for logits.
    input_size = config.data["input_size"]
    train_size = config.data["train_size"]
    MODEL_SCALE = config.data["model_scale"]
    
    # get center
    pred = hm > thresh
    pred_center = np.where(hm>thresh)
    
    # get regressions
    pred_r = regr[:,pred].T
    pred_rwh = regr_wh[:,pred].T

    # wrap as boxes
    # [xmin, ymin, width, height]
    # size as original image.
    boxes = []
    scores = hm[pred]
    for i, (b, wh) in enumerate(zip(pred_r, pred_rwh)):
        b /= MODEL_SCALE
        arr = np.array([pred_center[1][i]*MODEL_SCALE-b[0]*input_size/2 - wh[0]*input_size,
                        pred_center[0][i]*MODEL_SCALE-b[1]*input_size/2 - wh[1]*input_size, 
                      b[0]*input_size, b[1]*input_size])
        arr = np.clip(arr, 0, input_size)
        boxes.append(arr)
    return np.asarray(boxes), scores

# Wrapped heatmap function
def make_hm_regr(config, target, pass_center=False, train=False):
    if not train:
        input_size = config.data["input_size"]
        train_size = config.data["train_size"]
    else:
        input_size = config.data["train_size"]
        train_size = config.data["train_size"]
    #IN_SCALE = input_size/train_size
    IN_SCALE = 1
    MODEL_SCALE = config.data["model_scale"]
    
    # make output heatmap for single class
    hm = np.zeros([input_size//MODEL_SCALE, input_size//MODEL_SCALE])
    # make regr heatmap 
    regr = np.zeros([2, input_size//MODEL_SCALE, input_size//MODEL_SCALE])
    regr_wh = np.zeros([2, input_size//MODEL_SCALE, input_size//MODEL_SCALE])
    
    if len(target) == 0:
        return hm, regr, regr_wh
    
    if not pass_center:
        try:
            center = np.array([target["x"]+target["w"]/2, target["y"]+target["h"]/2, 
                           target["w"], target["h"]
                          ]).T
        except:
            center = np.array([int(target["x"]+target["w"]/2), int(target["y"]+target["h"]/2), 
                           int(target["w"]), int(target["h"])
                          ]).T.reshape(1,4)
    else:
        center = target
    
    # make a center point
    if config.data["hm"] == "gaussian":
        for c in center:
            hm = draw_msra_gaussian(hm, [int(c[0]/MODEL_SCALE/IN_SCALE), int(c[1]/MODEL_SCALE/IN_SCALE)], 
                                    sigma=np.clip(c[2]*c[3]//2000, 1, 3))    

    # convert targets to its center.
    regrs = center[:, 2:]/input_size/IN_SCALE*MODEL_SCALE

    # plot regr values to mask
    for r, c in zip(regrs, center):
        for i in range(-1, 2):
            for j in range(-1, 2):
                try:
                    if i==0:
                        ival=1.0
                    else:
                        ival=3/4
                    if j==0:
                        jval=1.0
                    else:
                        jval=3/4
                        
                    regr[:, int(c[0]/MODEL_SCALE/IN_SCALE+i), 
                         int(c[1]/MODEL_SCALE/IN_SCALE+j)] = r
                    
                    if config.data["hm"] == "3x3":
                        hm[int(c[1]/MODEL_SCALE/IN_SCALE+i), 
                             int(c[0]/MODEL_SCALE/IN_SCALE+j)] = ival*jval
                        
                    regr_wh[:, int(c[0]/MODEL_SCALE/IN_SCALE+i), 
                         int(c[1]/MODEL_SCALE/IN_SCALE+j)] \
                    = np.array([(-c[0]/MODEL_SCALE/IN_SCALE+int(c[0]/MODEL_SCALE/IN_SCALE))/input_size*MODEL_SCALE+i/input_size/IN_SCALE*MODEL_SCALE, 
                               (-c[1]/MODEL_SCALE/IN_SCALE+int(c[1]/MODEL_SCALE/IN_SCALE))/input_size*MODEL_SCALE+j/input_size/IN_SCALE*MODEL_SCALE])
                
                except:
                    pass
                
    regr[0] = regr[0].T; regr[1] = regr[1].T;
    regr_wh[0] = regr_wh[0].T; regr_wh[1] = regr_wh[1].T;
    return hm, regr, regr_wh

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

from nms import nms
def showbox(cfg, img, hm, regr, regr_wh, thresh=0.9, runnms=False, nmsthresh=0.45):
    boxes, scores = pred2box(cfg, hm, regr, regr_wh, thresh=thresh)
    print("preds prenms:",boxes.shape)
    if runnms:
        keep, count = nms(boxes, scores, nmsthresh, 1000)
        boxes = boxes[keep[:count]]
        print("preds postnms:", count)
    
    sample = img

    for box in boxes:
        # upper-left, lower-right
        cv2.rectangle(sample,
                      (int(box[0]), int(box[1]+box[3])),
                      (int(box[0]+box[2]), int(box[1])),
                      (220, 0, 0), 2)
    return sample

def showgtbox(cfg, img, hm, regr, regr_wh, thresh=0.9):
    boxes, _ = pred2box(cfg, hm, regr, regr_wh, thresh=thresh)
    print("GT boxes:", boxes.shape)
    sample = img

    for box in boxes:
        cv2.rectangle(sample,
                      (int(box[0]), int(box[1]+box[3])),
                      (int(box[0]+box[2]), int(box[1])),
                      (0, 220, 0), 3)
    return sample
