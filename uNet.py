import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as tFun
import numpy as np
import math
    
# Function to calculate Jaccard Index
def jaccardIndex(pred, target, train=False):
    pred = pred.detach().cpu().numpy().astype(bool) #Converts the prediction tensor to a boolean numpy array

    if train:
        target = target.argmax(dim=1) #Find the higher probability class per each pixel in the target tensor
    target = target.detach().cpu().numpy().astype(bool) #Converts the label tensor to a boolean numpy array
    
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou_score = np.sum(intersection) / np.sum(union)
    
    return iou_score

# Define a new combined loss function
def combinedLoss(pred, target, n_classes):
    criterion = nn.CrossEntropyLoss()
    ceLoss = criterion(pred, target)
    
    diceLossVal = diceLoss(tFun.softmax(pred, dim=1).float(), tFun.one_hot(target, n_classes).permute(0, 3, 1, 2).float(), multiclass=True)
    
    pred_mask = pred.argmax(dim=1)
    jaccard = jaccardIndex(pred_mask, target)
    jaccardLoss = 1 - jaccard  # Since we want to maximize Jaccard, take (1 - Jaccard) as the loss
    
    totalLoss = ceLoss + diceLossVal + jaccardLoss
    return totalLoss, ceLoss, diceLossVal, jaccardLoss

def diceCoeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon : float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclassDiceCoeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return diceCoeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def diceLoss(input: Tensor, target: Tensor, multiclass: bool = False):
    fn = multiclassDiceCoeff if multiclass else diceCoeff
    return 1 - fn(input, target, reduce_batch_first=True)

#All there classes represents the UNET architecture implementation
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, attention=False, residual=False):
        super().__init__()
        self.residual = residual
        self.attention = attention
        if residual:
            self.resconv = nn.Conv2d(in_channels, out_channels, 1, 1)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),  #Added
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if attention:
            self.spatialAtt = SpatialAttention()
            self.channelAtt = ChannelAttention(out_channels)
    def forward(self, x):
        if self.residual:
            residual = self.resconv(x)
        x = self.double_conv(x)
        if self.attention:
            channelAttVal = self.channelAtt(x)
            spatialAttVal = self.spatialAtt(x)
            channelAttValX = torch.mul(x, channelAttVal)
            spatialAttValX = torch.mul(x, spatialAttVal)
        if self.residual and self.attention:
            return channelAttValX + spatialAttValX + residual
        if self.residual and not self.attention:
            return x + residual
        if not self.residual and self.attention:
            return channelAttValX + spatialAttValX
        if not self.residual and not self.attention:
            return x
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, residual=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, attention=attention, residual=residual)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, residual=False, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels //2, attention=attention, residual=residual)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, attention=attention, residual=residual)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = tFun.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY //2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#Use UNet + Residual + Attention
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, residual=False,attention=False, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
        self.residual = residual

        self.inc = (DoubleConv(n_channels, 64, attention=attention, residual=residual))
        self.down1 = (Down(64, 128, attention,residual))
        self.down2 = (Down(128, 256, attention, residual))
        self.down3 = (Down(256, 512, attention, residual))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, attention,residual))
        self.up1 = (Up(1024, 512 // factor, attention, residual, bilinear))
        self.up2 = (Up(512, 256 // factor, attention, residual, bilinear))
        self.up3 = (Up(256, 128 // factor, attention, residual, bilinear))
        self.up4 = (Up(128, 64, attention, residual, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

# Is used to improve the netork by focalizyng on important parts of the image
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1) #The 2 input channels are for the mean and the maximum of each input channel
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True) #Calculate the average of each channel
        maxVal, _ = torch.max(x, dim=1, keepdim=True) #Calculate the maximum of each channel
        cat = torch.cat((avg, maxVal), dim=1) #Concatenate the average and the maximum
        x = self.conv(cat) #Apply the convolution defined previously
        return torch.sigmoid(x)

# It focalyzes on the most important channels of the image with regard to the features
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convBlock = nn.Sequential( #It is a block of 3 convolutions where the channel size is reduced by 16
            nn.Conv2d(in_channels, math.ceil(in_channels//16), 1),
            nn.Conv2d(math.ceil(in_channels//16), math.ceil(in_channels//16), 3, padding=1),
            nn.Conv2d(math.ceil(in_channels//16), 1, 1)
        )
    def forward(self, x):
        avg = torch.mean(x, dim=(-1, -2), keepdim=True)
        maxVal, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)
        maxVal = maxVal.view(x.size(0), x.size(1), 1, 1)
        outAvg = self.convBlock(avg)
        outMax = self.convBlock(maxVal)
        x = torch.add(outAvg, outMax)
        return torch.sigmoid(x)