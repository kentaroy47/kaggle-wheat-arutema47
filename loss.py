# focal loss
import torch

def neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pred = pred.unsqueeze(1).float()
  gt = gt.unsqueeze(1).float()

  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()
  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_inds
  neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, 3) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  #print(gt_regr.size())
  mask = mask.sum(1).unsqueeze(1).expand_as(gt_regr)
  #print(mask.size())

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss
  
def centerloss(prediction, mask, regr, weight=1, size_average=True, focal=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    if focal:
        mask_loss = neg_loss(pred_mask, mask)
    else:
        mask_loss = mask * \
                torch.log(pred_mask + 1e-12) + (1 - mask) * \
                torch.log(1 - pred_mask + 1e-12)
        mask_loss = -mask_loss.mean()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:3]
    a = False
    if regr.size()[1]>2:
        a = True
        regr_wh = regr[:, 2:]
        pred_regr = prediction[:, 3:]
        regr_loss2 = (torch.abs(pred_regr - regr_wh).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
        
    regr = regr[:, :2]
    pred_regr = prediction[:, 1:3]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    if a:
        regr_loss += regr_loss2.mean(0)*0.1
    
    
  
    # Sum
    regr_loss = regr_loss*weight
    loss = mask_loss +regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss ,mask_loss , regr_loss
  