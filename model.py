import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import pretrainedmodels
import timm
from torchvision import models

class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)

    def forward(self, x):
        return self.block(x)


class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
                               bias=False)

    def forward(self, x):
        return self.block(x)
    
# https://github.com/kentaroy47/efficientdet.pytorch/blob/master/BiFPN.py
class BiFPN(nn.Module):
    def __init__(self, num_channels):
        super(BiFPN, self).__init__()
        self.num_channels = num_channels
        out_channels = num_channels
        #self.conv7up = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),nn.BatchNorm2d(num_features=out_channels),nn.ReLU())
        self.conv6up = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels), nn.BatchNorm2d(num_features=out_channels),nn.ReLU())
        self.conv5up = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels), nn.BatchNorm2d(num_features=out_channels),nn.ReLU())
        self.conv4up = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels), nn.BatchNorm2d(num_features=out_channels),nn.ReLU())
        self.conv3up = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels), nn.BatchNorm2d(num_features=out_channels),nn.ReLU())
        self.conv4dw = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels), nn.BatchNorm2d(num_features=out_channels),nn.ReLU())
        self.conv5dw = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels), nn.BatchNorm2d(num_features=out_channels),nn.ReLU())
        self.conv6dw = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels), nn.BatchNorm2d(num_features=out_channels),nn.ReLU())
        #self.conv7dw = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels), nn.BatchNorm2d(num_features=out_channels),nn.ReLU())
        
    def forward(self, inputs):
        num_channels = self.num_channels
        P3_in, P4_in, P5_in, P6_in = inputs
        
        # upsample network
        #P7_up = self.conv7up(P7_in)
        #scale = (P6_in.size(3)/P7_up.size(3))        
        P6_up = self.conv6up(P6_in)
        scale = (P5_in.size(3)/P6_up.size(3))
        P5_up = self.conv5up(P5_in+self.Resize(scale_factor=scale)(P6_up))
        scale = (P4_in.size(3)/P5_up.size(3))
        P4_up = self.conv4up(P4_in+self.Resize(scale_factor=scale)(P5_up))
        scale = (P3_in.size(3)/P4_up.size(3))
        P3_out = self.conv3up(P3_in+self.Resize(scale_factor=scale)(P4_up))

        # fix to downsample by interpolation
        # downsample networks
        P4_out = self.conv4dw(P4_in + P4_up+F.interpolate(P3_out, P4_up.size()[2:]))
        P5_out = self.conv5dw(P5_in + P5_up+F.interpolate(P4_out, P5_up.size()[2:]))
        P6_out = self.conv6dw(P6_in + P6_up+F.interpolate(P5_out, P6_up.size()[2:]))
        #P7_out = self.conv7dw(P7_in + P7_up+F.interpolate(P6_out, P7_up.size()[2:]))
        return P3_out, P4_out, P5_out, P6_out #, P7_out

    @staticmethod
    def Conv(in_channels, out_channels, kernel_size, stride, padding, groups=1):
        features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        return features 
    @staticmethod
    def Resize(scale_factor=2, mode='bilinear'):
        upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsample
    
class EffFPN(nn.Module):
    """Feature Pyramid Network (FPN): top-down architecture with lateral
       connections. Can be used as feature extractor for object detection
       or segmentation.
    """

    def __init__(self, slug, num_filters=256, pretrained=True, bifpn=False):
        """Creates an `FPN` instance for feature extraction.
        Args:
          slug: model slug e.g. 'r18', 'r101' for ResNet
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
          num_input_channels: number fo input channels
        """
        self.slug = slug

        super().__init__()
        if not pretrained:
            print("Caution, not loading pretrained weights.")

        if slug == "eff5":
            basemodel = timm.create_model('tf_efficientnet_b5_ns', pretrained=pretrained)
            num_bottleneck_filters = 512
        else:
            assert False, "Bad slug: %s" % slug
        
        self.bifpn = bifpn
        if bifpn:
            self.BiFPN = BiFPN(num_filters)
        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.lateral4 = Conv1x1(num_bottleneck_filters, num_filters)
        self.lateral3 = Conv1x1(176, num_filters)
        self.lateral2 = Conv1x1(64, num_filters)
        self.lateral1 = Conv1x1(40, num_filters)

        self.smooth4 = Conv3x3(num_filters, num_filters)
        self.smooth3 = Conv3x3(num_filters, num_filters)
        self.smooth2 = Conv3x3(num_filters, num_filters)
        self.smooth1 = Conv3x3(num_filters, num_filters)
        
        self.enc1 = nn.Sequential(basemodel.blocks[0:2])
        self.enc2 = nn.Sequential(basemodel.blocks[2:3])
        self.enc3 = nn.Sequential(basemodel.blocks[3:5])
        self.enc4 = nn.Sequential(basemodel.blocks[5:7])
        
        self.enc0 = nn.Sequential(basemodel.conv_stem, basemodel.bn1, basemodel.act1)

    def forward_s4(self, enc0):
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)

        # Top-down pathway
        if not self.bifpn:
            map4 = lateral4 # 16x16
            map3 = lateral3 + nn.functional.interpolate(map4, scale_factor=2,
                                                        mode="nearest") # 32x32
            map2 = lateral2 + nn.functional.interpolate(map3, scale_factor=2,
                                                        mode="nearest")
            map1 = lateral1 + nn.functional.interpolate(map2, scale_factor=2,
                                                        mode="nearest")
        else:
            map1, map2, map3, map4 = self.BiFPN([lateral1,lateral2,lateral3,lateral4,])
        # Reduce aliasing effect of upsampling

        map4 = self.smooth4(map4)
        map3 = self.smooth3(map3)
        map2 = self.smooth2(map2)
        map1 = self.smooth1(map1)

        return map1, map2, map3, map4

    def forward(self, x):
        # Bottom-up pathway, from ResNet

        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"
        enc0 = self.enc0(x)

        map1, map2, map3, map4 = self.forward_s4(enc0)
        return enc0, map1, map2, map3, map4


    
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)
        
    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2))
        else:
            x = x1
        x = self.conv(x)
        return x

# FPN implementation from 
# https://github.com/bamps53/kaggle-autonomous-driving2019/blob/master/models/centernet.py
class FPN(nn.Module):
    """Feature Pyramid Network (FPN): top-down architecture with lateral
       connections. Can be used as feature extractor for object detection
       or segmentation.
    """

    def __init__(self, slug, num_filters=256, pretrained=True, bifpn=False):
        """Creates an `FPN` instance for feature extraction.
        Args:
          slug: model slug e.g. 'r18', 'r101' for ResNet
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
          num_input_channels: number fo input channels
        """
        self.slug = slug

        super().__init__()
        if not pretrained:
            print("Caution, not loading pretrained weights.")

        if slug == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'resnet152':
            self.resnet = models.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx101':
            self.resnet = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
            num_bottleneck_filters = 2048
        elif slug == "rx102":
            self.resnet = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
            num_bottleneck_filters = 2048
        elif slug == "seresnext":
            self.resnet = pretrainedmodels.se_resnext50_32x4d(pretrained="imagenet")
            num_bottleneck_filters = 2048
        else:
            assert False, "Bad slug: %s" % slug

        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392
        self.bifpn = bifpn
        if bifpn:
            self.BiFPN = BiFPN(num_filters)
        
        self.lateral4 = Conv1x1(num_bottleneck_filters, num_filters)
        self.lateral3 = Conv1x1(num_bottleneck_filters // 2, num_filters)
        self.lateral2 = Conv1x1(num_bottleneck_filters // 4, num_filters)
        self.lateral1 = Conv1x1(num_bottleneck_filters // 8, num_filters)

        self.smooth4 = Conv3x3(num_filters, num_filters)
        self.smooth3 = Conv3x3(num_filters, num_filters)
        self.smooth2 = Conv3x3(num_filters, num_filters)
        self.smooth1 = Conv3x3(num_filters, num_filters)

    def forward_s4(self, enc0):
        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)

        # Top-down pathway
        if not self.bifpn:
            map4 = lateral4
            map3 = lateral3 + nn.functional.interpolate(map4, scale_factor=2,
                                                        mode="nearest")
            map2 = lateral2 + nn.functional.interpolate(map3, scale_factor=2,
                                                        mode="nearest")
            map1 = lateral1 + nn.functional.interpolate(map2, scale_factor=2,
                                                        mode="nearest")
        else:
            map1, map2, map3, map4 = self.BiFPN([lateral1,lateral2,lateral3,lateral4,])
        # Reduce aliasing effect of upsampling

        map4 = self.smooth4(map4)
        map3 = self.smooth3(map3)
        map2 = self.smooth2(map2)
        map1 = self.smooth1(map1)

        return map1, map2, map3, map4

    def forward(self, x):
        # Bottom-up pathway, from ResNet

        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"
        if 'serx' in self.slug:
            enc0 = self.resnet.layer0(x)
        else:
            enc0 = self.resnet.conv1(x)
            enc0 = self.resnet.bn1(enc0)
            enc0 = self.resnet.relu(enc0)
            enc0 = self.resnet.maxpool(enc0)

        map1, map2, map3, map4 = self.forward_s4(enc0)
        return enc0, map1, map2, map3, map4
    
class CenterNetFPN(nn.Module):
    """Semantic segmentation model on top of a Feature Pyramid Network (FPN).
    """

    def __init__(self, slug, num_classes=2, num_filters=128,
                 num_filters_fpn=256, upconv=False, pretrained=True, bifpn=False):
        """Creates an `FPNSegmentation` instance for feature extraction.
        Args:
          slug: model slug e.g. 'r18', 'r101' for ResNet
          num_classes: number of classes to predict
          num_filters: the number of filters in each segmentation head pyramid
                       level
          num_filters_fpn: the number of filters in each FPN output pyramid
                           level
          pretrained: use ImageNet pre-trained backbone feature extractor
          num_input_channels: number of input channels e.g. 3 for RGB
          output_size. Tuple[int, int] height, width
        """

        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        if "eff" in slug:
            self.fpn = EffFPN(slug=slug, num_filters=num_filters_fpn,
                       pretrained=pretrained, bifpn=bifpn)
        else:
            self.fpn = FPN(slug=slug, num_filters=num_filters_fpn,
                       pretrained=pretrained, bifpn=bifpn)
        # The segmentation heads on top of the FPN

        self.head1 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
                                   Conv3x3(num_filters, num_filters))
        self.head2 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
                                   Conv3x3(num_filters, num_filters))
        self.head3 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
                                   Conv3x3(num_filters, num_filters))
        self.head4 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
                                   Conv3x3(num_filters, num_filters))

        self.hm = nn.Conv2d(4 * num_filters, 1, 3, padding=1)

        self.classes_embedding = nn.Sequential(
            nn.Conv2d(4 * num_filters, 4 * num_filters, 3, padding=1),
            nn.ReLU(inplace=True))

        self.classes = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(4 * num_filters, num_classes, 1)
        )

        if upconv:
            self.up8 = nn.ConvTranspose2d(
                num_filters, num_filters, 8, stride=8)
            self.up4 = nn.ConvTranspose2d(
                num_filters, num_filters, 4, stride=4)
            self.up2 = nn.ConvTranspose2d(
                num_filters, num_filters, 2, stride=2)
        else:
            self.up8 = torch.nn.Upsample(scale_factor=8, mode='nearest')
            self.up4 = torch.nn.Upsample(scale_factor=4, mode='nearest')
            self.up2 = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def gather_embeddings(self, embeddings, centers):
        gathered_embeddings = []
        for sample_index in range(len(centers)):
            center_mask = centers[sample_index, :, 0] != -1
            if center_mask.sum().item() == 0:
                continue
            per_sample_centers = centers[sample_index][center_mask]
            emb = embeddings[sample_index][:, per_sample_centers[:, 1],
                                           per_sample_centers[:, 0]].transpose(0, 1)
            gathered_embeddings.append(emb)
        gathered_embeddings = torch.cat(gathered_embeddings, 0)

        return gathered_embeddings

    def forward(self, x, centers=None, return_embeddings=False):
        # normalize
        #x = x / 127.5 - 1.0
        enc0, map1, map2, map3, map4 = self.fpn(x)

        h4 = self.head4(map4)
        h3 = self.head3(map3)
        h2 = self.head2(map2)
        h1 = self.head1(map1)

        map4 = self.up8(h4)
        map3 = self.up4(h3)
        map2 = self.up2(h2)
        map1 = h1

        final_map = torch.cat([map4, map3, map2, map1], 1)
        hm = self.hm(final_map)
        classes_embedding = self.classes_embedding(final_map)
        if return_embeddings:
            return hm, classes_embedding

        if centers is not None:
            gathered_embeddings = self.gather_embeddings(classes_embedding,
                                                         centers)
            classes = self.classes(gathered_embeddings.unsqueeze(
                -1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        else:
            classes = self.classes(classes_embedding)

        return hm, classes
    
class centernet(nn.Module):
    def __init__(self, n_classes=2, model_name="resnet18", MODEL_SCALE=4):
        super(centernet, self).__init__()
        self.MODEL_SCALE = MODEL_SCALE
        # create backbone.
        basemodel = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        basemodel = nn.Sequential(*list(basemodel.children())[:-2])
        # set basemodel
        self.base_model = basemodel
        
        if model_name == "resnet34" or model_name=="resnet18":
            num_ch = 512
        else:
            num_ch = 2048
        
        self.up1 = up(num_ch, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 256)
        # output classification
        self.outc = nn.Conv2d(256, 1, 1)
        # output residue
        self.outr = nn.Conv2d(256, n_classes, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.base_model(x)
        
        # Add positional info        
        x = self.up1(x)
        x = self.up2(x)
        if self.MODEL_SCALE == 4:
            x = self.up3(x)
        outc = self.outc(x)
        outr = self.outr(x)
        return outc, outr