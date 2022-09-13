from torchvision import models

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class DeepLabModel(torch.nn.Module):
    def __init__(self, pretrained: bool=True, n_classes: int=44, device="cpu"):
        super().__init__()
        self.device = device
        self.n_classes = n_classes

        self.model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
            weights_backbone=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        ).to(device)

        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x):
        x = self.model(x)['out']
        x[:, :21] = torch.sigmoid(x[:, :21])
        return x

if __name__ == "__main__":

    with torch.no_grad():
        testin = torch.randn(1, 3, 512, 512, device="cuda")
        # testin = torch.randn(1, 3, 1213, 956, device="cuda")
        model = DeepLabModel(n_classes=44, device="cuda")
        model.cuda()
        model.eval()
        ### Shared VGG encoder
        pred = model(testin)
        # 0: 64x256x256, 1: 128x128x128
        # 2: 256x64x64, 3: 512x32x32, 4: 512x16x16
        print(pred.shape)
