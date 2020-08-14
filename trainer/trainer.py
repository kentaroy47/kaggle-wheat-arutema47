import pandas as pd
from metric import calculate_precision, calculate_image_precision
from dataset.dataset import pred2box
import torch
import gc
import os
from tqdm import tqdm_notebook as tqdm
import numpy as np
from nms import nms

def centernet_eval(model, val_loader, config, criterion, optimizer, exp_lr_scheduler, line=None, logs_eval=None, epoch=0, map=False, threshs=[0.3], save_stride=2):
    iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
    running_loss = 0.0
    running_mask = 0.0
    running_regr = 0.0
    
    precision05 = []
    precisions = []
    thresh = 0.3
    model.eval()
    watermark = config["watermark"]
    input_size = config.data["input_size"]
    train_size = config.data["train_size"]
    REGR_SCALE = config.train["regr_scale"]
    device = config["device"]
    
    t = tqdm(val_loader)
    for idx, (img, hm_gts, regr_gts) in enumerate(t):
        # send to gpu
        img = img.to(device)
        hm_gts = hm_gts.to(device)
        regr_gts = regr_gts.to(device)

        # predict
        with torch.no_grad():
            hms, regrs = model(img.to(device).float())
        # loss
        regrs = regrs /  (input_size/train_size)
        preds = torch.cat((hms, regrs), 1)
        loss, mask_loss, regr_loss = criterion(preds, hm_gts, regr_gts, REGR_SCALE)
        # misc
        running_loss += loss
        running_mask += mask_loss
        running_regr += regr_loss
        
        t.set_description(f't (l={running_loss/(idx+1):.3f})(m={running_mask/(idx+1):.4f})(r={running_regr/(idx+1):.4f})')
        if map:
            # calculate mAP
            hm_gts = hm_gts.cpu().numpy()
            regr_gts = regr_gts.cpu().numpy()
            for hm, regr, hm_gt, regr_gt  in zip(hms, regrs, hm_gts, regr_gts):
                # process predictions
                hm = hm.cpu().numpy().squeeze(0)
                regr = regr.cpu().numpy() 
                hm = torch.sigmoid(torch.from_numpy(hm)).numpy()

                boxes, scores = pred2box(config,hm, regr[0:2], regr[2:], thresh)
                boxes_gt, scores_gt = pred2box(config,hm_gt, regr_gt[:2], regr_gt[2:], 0.99)

                # Filter by nms
                keep, count = nms(boxes, scores)
                boxes = boxes[keep[:count]]
                scores = scores[keep[:count]]

                preds_sorted_idx = np.argsort(scores)[::-1]
                boxes_sorted = boxes[preds_sorted_idx]

                precision, fn_boxes, fp_boxes = calculate_precision(boxes_sorted, boxes_gt, threshold=0.5, form='coco')
                precision05.append(precision)

                image_precision = calculate_image_precision(boxes_sorted, boxes_gt,
                                                    thresholds=iou_thresholds,
                                                    form='coco', debug=False)
                precisions.append(image_precision)
            
    print("threshold = ", thresh)
    print("mAP at threshold 0.5: {0:.4f}".format(np.mean(precision05)))
    print("mAP at threshold 0.5:0.75: {0:.4f}".format(np.mean(precisions)))
    
    print('val loss : {:.4f}'.format(running_loss/len(val_loader)))
    print('maskloss : {:.4f}'.format(running_mask/(len(val_loader))))
    print('regrloss : {:.4f}'.format(running_regr/(len(val_loader))))
    
    exp_lr_scheduler.step(running_loss)
    
    # save logs
    log_epoch = {'epoch': epoch+1, 'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                    'mAP50': np.mean(precision05), "map50:75": np.mean(precisions),
                 'loss': running_loss/len(val_loader), "mask": running_mask/(len(val_loader)), 
                 "regr": running_regr/(len(val_loader))}
    # 
    if logs_eval is not None:
        logs_eval.append(log_epoch)
        df = pd.DataFrame(logs_eval)
        df.to_csv("log/log_output_eval_{}_{}.csv".format(watermark, input_size))
    
    if line is not None:
        line.send('epoch {}, mAP50 {}, mAP50:75 {}, lr {}'.format(epoch,  np.mean(precision05), np.mean(precisions), optimizer.state_dict()['param_groups'][0]['lr']))
        
    # GC
    torch.cuda.empty_cache()
    gc.collect()
    
    if epoch%save_stride==0:
        torch.save(model.state_dict(), './models/{}_{}_{}_{}.pth'.format(watermark, epoch, input_size, train_size))
        
    return np.mean(precision05), np.mean(precisions)


def centernet_train(model, train_loader, config, criterion, optimizer, logs=None, epoch=0):
    if logs is None:
        logs = []
    model.eval()
    ###########
    running_loss = 0.0
    running_mask = 0.0
    running_regr = 0.0
    t = tqdm(train_loader)
    watermark = config["watermark"]
    input_size = config.data["input_size"]
    REGR_SCALE = config.train["regr_scale"]
    device = config["device"]
    #############
    # set opt
    optimizer.zero_grad()
    for idx, (img, hm, regr) in enumerate(t):       
        # send to gpu
        img = img.to(device)
        hm_gt = hm.to(device)
        regr_gt = regr.to(device)
               
        # run model
        hm, regr = model(img.float())
        preds = torch.cat((hm, regr), 1)
            
        loss, mask_loss, regr_loss = criterion(preds, hm_gt, regr_gt, REGR_SCALE)
        # misc
        running_loss += loss
        running_mask += mask_loss
        running_regr += regr_loss
        loss.backward()
        
        if (idx+1) % config.train["accumulation_size"] == 0:           
            optimizer.step()
            optimizer.zero_grad()
        
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
    df.to_csv("log/log_output_train_{}_{}.csv".format(watermark, input_size))
    # GC
    torch.cuda.empty_cache()
    gc.collect()