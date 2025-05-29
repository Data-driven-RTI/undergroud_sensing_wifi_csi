import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models._utils import IntermediateLayerGetter


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        num = predictions.size(0)
        pred_flat = predictions.view(num, -1)
        targ_flat = targets.view(num, -1).float()
        intersection = (pred_flat * targ_flat).sum(1)
        pred_sum = pred_flat.sum(1)
        targ_sum = targ_flat.sum(1)
        dice_coeff = (2. * intersection + self.epsilon) / (pred_sum + targ_sum + self.epsilon)
        dice_loss = 1. - dice_coeff.mean()
        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_loss = nn.BCELoss() 
        self.dice_loss = DiceLoss(epsilon=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):
        targets = targets.float() 
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        combined_loss = (self.bce_weight * bce) + (self.dice_weight * dice)
        return combined_loss


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(residual) 
        out = self.relu(out)
        return out


class UpsamplePixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1, bias=False)
        self.ps = nn.PixelShuffle(scale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)  
        x1 = self.W_x(x)  

        psi = self.relu(g1 + x1) 
        psi = self.psi(psi)      

        x_att = x * psi          

        return x_att


class SignalToMaskUNet(nn.Module):
    def __init__(self, input_channels, base_filters=64, output_size=(360, 360)):
        super().__init__()
        self.input_channels = input_channels
        self.output_size = output_size
        bf = base_filters

        self.enc1_conv = ResConvBlock(input_channels, bf) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.enc2_conv = ResConvBlock(bf, bf * 2)         
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.enc3_conv = ResConvBlock(bf * 2, bf * 4)     
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.bottleneck = ResConvBlock(bf * 4, bf * 8)    

        self.up1 = UpsamplePixelShuffleBlock(bf * 8, bf * 4) 
        self.att1 = AttentionBlock(F_g=bf * 4, F_x=bf * 4, F_int=bf * 2)
        self.dec1_conv = ResConvBlock(bf * 4 + bf * 4, bf * 4) 

        self.up2 = UpsamplePixelShuffleBlock(bf * 4, bf * 2) 
        self.att2 = AttentionBlock(F_g=bf * 2, F_x=bf * 2, F_int=bf)
        self.dec2_conv = ResConvBlock(bf * 2 + bf * 2, bf * 2) 

        self.up3 = UpsamplePixelShuffleBlock(bf * 2, bf)     
        self.att3 = AttentionBlock(F_g=bf, F_x=bf, F_int=bf // 2)
        self.dec3_conv = ResConvBlock(bf + bf, bf)           

        self.up4 = UpsamplePixelShuffleBlock(bf, bf // 2)
        self.dec4_conv = ResConvBlock(bf // 2, bf // 2) 

        self.up5 = UpsamplePixelShuffleBlock(bf // 2, bf // 4)
        self.dec5_conv = ResConvBlock(bf // 4, bf // 4)

        self.up6 = UpsamplePixelShuffleBlock(bf // 4, bf // 8)
        self.dec6_conv = ResConvBlock(bf // 8, bf // 8)

        self.up7 = UpsamplePixelShuffleBlock(bf // 8, bf // 16)
        self.dec7_conv = ResConvBlock(bf // 16, bf // 16)

        self.final_resize = nn.Upsample(size=output_size, mode='bilinear', align_corners=False)
        self.final_refine = ResConvBlock(bf // 16, bf // 16)
        self.final_conv = nn.Conv2d(bf // 16, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()


    def forward(self, x):
        e1 = self.enc1_conv(x)      
        p1 = self.pool1(e1)         

        e2 = self.enc2_conv(p1)     
        p2 = self.pool2(e2)         

        e3 = self.enc3_conv(p2)     
        p3 = self.pool3(e3)         

        b = self.bottleneck(p3)     

        d1_up = self.up1(b)         
        e3_att = self.att1(g=d1_up, x=e3) 
        d1 = torch.cat([d1_up, e3_att], dim=1) 
        d1 = self.dec1_conv(d1)     

        d2_up = self.up2(d1)        
        e2_att = self.att2(g=d2_up, x=e2) 
        d2 = torch.cat([d2_up, e2_att], dim=1) 
        d2 = self.dec2_conv(d2)     

        d3_up = self.up3(d2)        
        e1_att = self.att3(g=d3_up, x=e1) 
        d3 = torch.cat([d3_up, e1_att], dim=1) 
        d3 = self.dec3_conv(d3)     

        d4_up = self.up4(d3)        
        d4 = self.dec4_conv(d4_up)  

        d5_up = self.up5(d4)        
        d5 = self.dec5_conv(d5_up)  

        d6_up = self.up6(d5)        
        d6 = self.dec6_conv(d6_up)  

        d7_up = self.up7(d6)        
        d7 = self.dec7_conv(d7_up)  

        resized = self.final_resize(d7) 
        refined = self.final_refine(resized) 
        logits = self.final_conv(refined)    
        output_mask = self.final_activation(logits)

        return output_mask


class CustomSignalBackbone(nn.Module):
    def __init__(self, input_channels, base_filters=64, output_channels=2048):
        super().__init__()
        bf = base_filters

        self.stage1 = nn.Sequential(
            ResConvBlock(input_channels, bf * 2),
            ResConvBlock(bf * 2, bf * 4)
        )
        self.stage2 = nn.Sequential(
            ResConvBlock(bf * 4, bf * 8, stride=2),
            ResConvBlock(bf * 8, bf * 8)
        )
        self.stage3 = nn.Sequential(
            ResConvBlock(bf * 8, bf * 16, stride=2),
            ResConvBlock(bf * 16, bf * 16)
        )
        self.stage4 = nn.Sequential(
             ResConvBlock(bf * 16, output_channels // 2),
             nn.Conv2d(output_channels // 2, output_channels, kernel_size=3, padding=2, dilation=2, bias=False),
             nn.BatchNorm2d(output_channels),
             nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x) 
        return x 


def create_final_upsampler(in_channels, output_size=(360, 360)):
    layers = []
    current_channels = in_channels 
    upsample_channels = [16, 32, 64, 32, 16] 

    if current_channels == 1:
         layers.append(nn.Conv2d(1, upsample_channels[0], kernel_size=3, padding=1))
         layers.append(nn.ReLU(inplace=True))
         current_channels = upsample_channels[0]

    for i in range(min(6, len(upsample_channels) -1 )):
         next_channels = upsample_channels[i+1]
         layers.append(UpsamplePixelShuffleBlock(current_channels, next_channels))
         current_channels = next_channels

    layers.append(nn.Upsample(size=output_size, mode='bilinear', align_corners=False))
    layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=3, padding=1))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(current_channels // 2, 1, kernel_size=1))

    return nn.Sequential(*layers)


class AdaptedFCN(nn.Module):
    def __init__(self, input_channels, base_filters=64, output_size=(360, 360), backbone_output_channels=2048):
        super().__init__()
        self.output_size = output_size

        self.backbone = CustomSignalBackbone(input_channels, base_filters, backbone_output_channels)

        fcn_model = fcn_resnet50(weights=None, progress=False) 
        self.segmentation_head = fcn_model.classifier
        head_input_channels = 512 
        fcn_final_conv_in_channels = self.segmentation_head[-1].in_channels 

        if backbone_output_channels != head_input_channels:
             self.segmentation_head[0] = nn.Conv2d(backbone_output_channels, head_input_channels, kernel_size=3, padding=1, bias=False)

        self.segmentation_head[-1] = nn.Conv2d(fcn_final_conv_in_channels, 1, kernel_size=1)

        self.final_upsampler = create_final_upsampler(in_channels=1, output_size=output_size)

        self.final_activation = nn.Sigmoid()


    def forward(self, x):
        features = self.backbone(x)

        head_output = self.segmentation_head(features)

        logits = self.final_upsampler(head_output)

        output_mask = self.final_activation(logits)
        return output_mask


class AdaptedDeepLabV3(nn.Module):
    def __init__(self, input_channels, base_filters=64, output_size=(360, 360), backbone_output_channels=2048):
        super().__init__()
        self.output_size = output_size

        self.backbone = CustomSignalBackbone(input_channels, base_filters, backbone_output_channels)

        deeplab_model = deeplabv3_resnet50(weights=None, progress=False) 
        self.segmentation_head = deeplab_model.classifier 

        aspp_input_channels = 2048 
        if backbone_output_channels != aspp_input_channels:
            print(f"WARNING: Custom backbone output channels ({backbone_output_channels}) "
                  f"do not match default DeepLab Head input ({aspp_input_channels}). "
                  f"ASPP module might fail or need adaptation.")

        deeplab_final_conv_in_channels = self.segmentation_head[-1].in_channels 
        self.segmentation_head[-1] = nn.Conv2d(deeplab_final_conv_in_channels, 1, kernel_size=1)

        self.final_upsampler = create_final_upsampler(in_channels=1, output_size=output_size)

        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)

        head_output = self.segmentation_head(features)

        logits = self.final_upsampler(head_output)

        output_mask = self.final_activation(logits)
        return output_mask