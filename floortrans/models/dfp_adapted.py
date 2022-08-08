from torchvision import models
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DFPmodel(torch.nn.Module):
    """Model of the Deep FloorPlan Recognition [1].

    Receives images as inputs and outputs the pixel-wise classification concerning
    the different room regions and boundaries. 
    
    It first relies on a VGG-16 model to extract a shared feature vector, common
    to both tasks.
    The authors proposed a multi-task model comprising different components:

    1. **VGG-16 feature extractor**: extracts a shared feature vector from the
        floor plans, that will be shared by the following components of the network.

    2. **Room boundary prediction (RB)**, focused on predicting the elements that
        separate the room regions, like walls, windows, and doors. Implemented as
        a VGG-16 decoder.

    3. **Room type classification (RT)**, focused on predicting the room regions. 
        Implemented as a VGG-16 decoder.
    
    4. **Room-boundary-guided attention mechanism** that allows using the 
        room-boundary context features to bound and guide the discovery of room
        regions. Technically, the features are passed from the room boundary
        decoder to the room-type decoder.
    

    Parameters
    ----------
    pretrained: bool, defaults to true
        Whether to use VGG16's encoder pretrained weights.
    freeze: bool, defaults to true
        Whether to freeze the weights of the VGG-16 encoder model.

    number_of_classes: int, defaults to 44


    References
    ----------
    1 - https://arxiv.org/pdf/1908.11025.pdf
    """
    def __init__(self, pretrained: bool=True, freeze: bool=True, n_classes: int=44):
        super(DFPmodel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ----------------------------------------------------
        # 1. Initialize VGG encoder (component 1)
        # ----------------------------------------------------
        # Output will be shared with RB and RT
        self._initializeVGG(pretrained, freeze)

        # ----------------------------------------------------
        # 2. Room Boundary Prediction 
        # ----------------------------------------------------
        # Number of layers
        rblist = [512, 256, 128, 64, 32, 3]
        # ^Note: Last layer size equals number of RB classes: walls, doors, windows

        # *rbtrans*: VGG decoder that will be used to output the classes
        self.rbtrans = nn.ModuleList([self._transconv2d(
            rblist[i], rblist[i+1], 4, 2, 1) for i in range(len(rblist)-2)])
        
        # *rbconvs* and *rbgrs*: convolutions to use in the spatial contextual module.
        self.rbconvs = nn.ModuleList([self._conv2d(
            rblist[i], rblist[i+1], 3, 1, 1) for i in range(len(rblist)-1)])
        
        self.rbgrs = nn.ModuleList([self._conv2d(
            rblist[i], rblist[i], 3, 1, 1) for i in range(1, len(rblist)-1)])

        # ----------------------------------------------------
        # 3. Room Type Prediction 
        # ----------------------------------------------------
        rtlist = [512, 256, 128, 64, 32]

        # *rttrans*: VGG decoder used to output the room regions classes
        self.rttrans = nn.ModuleList([self._transconv2d(
            rtlist[i], rtlist[i+1], 4, 2, 1) for i in range(len(rtlist)-1)])
        
        # *rtconvs*: convolution used 
        self.rtconvs = nn.ModuleList([self._conv2d(
            rtlist[i], rtlist[i+1], 3, 1, 1) for i in range(len(rtlist)-1)])
        self.rtgrs = nn.ModuleList([self._conv2d(
            rtlist[i], rtlist[i], 3, 1, 1) for i in range(1, len(rtlist))])

        # ----------------------------------------------------
        # 4. Attention Mechanism 
        # ----------------------------------------------------
        # Attention Non-local context
        clist = [256, 128, 64, 32]
        self.ac1s = nn.ModuleList(self._conv2d(
            clist[i], clist[i], 3, 1, 1) for i in range(len(clist)))
        self.ac2s = nn.ModuleList(self._conv2d(
            clist[i], clist[i], 3, 1, 1) for i in range(len(clist)))
        self.ac3s = nn.ModuleList(self._conv2d(
            clist[i], 1, 1, 1) for i in range(len(clist)))

        self.xc1s = nn.ModuleList(self._conv2d(
            clist[i], clist[i], 3, 1, 1) for i in range(len(clist)))
        self.xc2s = nn.ModuleList(self._conv2d(
            clist[i], 1, 1, 1) for i in range(len(clist)))

        self.ecs = nn.ModuleList(self._conv2d(
            1, clist[i], 1, 1) for i in range(len(clist)))
        self.rcs = nn.ModuleList(self._conv2d(
            2*clist[i], clist[i], 1, 1) for i in range(len(clist)))
        
        # 4.1. Direction aware kernel
        # -----------------------------------------------------------
        # Each line will create a non-trainable conv layer with shape
        # 1 x 1 x dim x 1 or any of the other combinations specified
        dak = [9, 17, 33, 65] # TODO: why these numbers?
        
        # horizontal
        self.hs = nn.ModuleList(self._dirawareLayer([1, 1, dim, 1]) 
                for dim in dak)
        # vertical
        self.vs = nn.ModuleList(self._dirawareLayer([1, 1, 1, dim])
                for dim in dak)
        # diagonal
        self.ds = nn.ModuleList(self._dirawareLayer([1, 1, dim, dim],
            diag=True) for dim in dak)
        # diagonal flip
        self.dfs = nn.ModuleList(self._dirawareLayer([1, 1, dim, dim],
            diag=True, flip=True) for dim in dak)

        # Last layer # TODO: last layer of what?
        # - Where are we concatenating layers?
        self.last = self._conv2d(clist[-1], n_classes-3, 1, 1)  # TODO: Why n_classes-3?
        # original Torch implementation is conv2d(clist[-1], 9, 1, 1)
    
    def _dirawareLayer(self, shape: List[int], diag: bool=False, flip: bool=False, trainable: bool=False):
        # Create a non-trainable kernel for the direction-aware kernel
        w = self.constant_kernel(shape, diag=diag, flip=flip, trainable=trainable)
        # compute padding
        pad = ((np.array(shape[2:])-1)/2).astype(int)
        # 
        conv = nn.Conv2d(1, 1, shape[2:], 1, list(pad), bias=False)
        conv.weight = w
        return conv

    def _initializeVGG(self, pretrained: bool, freeze: bool):
        """Initialize the VGG encoder model."""
        encmodel = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)

        if freeze:
            for child in encmodel.children():
                for param in child.parameters():
                    param.requires_grad = False
        # Note: 
        # I printed the features and it consists of exactly 31 layers,
        # making the slicing irrelevant.
        # print("\n\n\n\n\n\n", encmodel.features, "\n\n\n\n\n\n")
        # https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg16
        features = list(encmodel.features)[:31] 
        self.features = nn.ModuleList(features)
    
    def _conv2d(self, in_: int, out: int, kernel: int, stride: int=1, padding: int=0):
        """Creates a 2D convolution and initializes using He initialization."""
        conv2d = nn.Conv2d(in_, out, kernel, stride, padding)
        nn.init.kaiming_uniform_(conv2d.weight)
        nn.init.zeros_(conv2d.bias)
        return conv2d

    def _transconv2d(self, in_, out, kernel, stride=1, padding=0):
        """Upsampling transformation and initializes its weights."""
        transconv2d = nn.ConvTranspose2d(in_, out, kernel, stride, padding)
        nn.init.kaiming_uniform_(transconv2d.weight)
        nn.init.zeros_(transconv2d.bias)
        return transconv2d

    def constant_kernel(self, shape, value=1, diag=False,
            flip=False, trainable=False):
        if not diag:
            k = nn.Parameter(torch.ones(shape)*value, requires_grad=trainable)
        else:
            w = torch.eye(shape[2], shape[3])
            if flip:
                w = torch.reshape(w, (1, shape[2], shape[3]))
                w = w.flip(0, 1)
            w = torch.reshape(w, shape)
            k = nn.Parameter(w, requires_grad=trainable)
        return k
    
    # Not used in the project. Let's simplify.
    # -------------------------------------------
    # def context_conv2d(self, t, dim=1, size=7, diag=False,
    #         flip=False, stride=1, trainable=False):
    #     N, C, H, W = t.size(0), t.size(1), t.size(2), t.size(3)
    #     in_dim = C
    #     size = size if isinstance(size, (tuple, list)) else [size, size]
    #     stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
    #     shape = [dim, in_dim, size[0], size[1]]
    #     w = self.constant_kernel(shape, diag=diag, flip=flip, trainable=trainable)
    #     pad = ((np.array(shape[2:])-1)/2).astype(int)
    #     conv = nn.Conv2d(1, 1, shape[2:], 1, list(pad), bias=False)
    #     conv.weight = w
    #     conv.to(self.device);
    #     return conv(t)

    def non_local_context(self, t1, t2, idx, stride=4):
        N, C, H, W = t1.size(0), t1.size(1), t1.size(2), t1.size(3)
        hs = H // stride if (H // stride) > 1 else (stride-1)
        vs = W // stride if (W // stride) > 1 else (stride-1)
        hs = hs if (hs%2!=0) else hs+1
        vs = hs if (vs%2!=0) else vs+1

        a = F.relu(self.ac1s[idx](t1))
        a = F.relu(self.ac2s[idx](a))
        a = torch.sigmoid(self.ac3s[idx](a))
        x = F.relu(self.xc1s[idx](t2))
        x = torch.sigmoid(self.xc2s[idx](x))
        x = a*x

        # direction-aware kernels
        h = self.hs[idx](x)
        v = self.vs[idx](x)
        d1 = self.ds[idx](x)
        d2 = self.dfs[idx](x)

        # double attention 
        c1 = a*(h+v+d1+d2)

        # expand channel
        c1 = self.ecs[idx](c1)

        # concatenation + upsample
        features = torch.cat((t2, c1), dim=1)

        out = F.relu(self.rcs[idx](features))
        return out

    def forward(self, x):
        # 21 heatmap classes
        # 12 room classes
        # room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
        # 11 icon classes
        # icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

        print("dfp.forward", x.shape)
        N, C, H, W = x.shape

        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 9, 16, 23, 30}:
                results.append(x)
        rbfeatures = []
        for i, rbtran in enumerate(self.rbtrans):
            residual = results[3-i]
            x = rbtran(x)

            diffH = residual.shape[2] - x.shape[2]
            diffW = residual.shape[3] - x.shape[3]
            x = F.pad(x, (diffW // 2, diffW - diffW//2,
                          diffH // 2, diffH - diffH//2))

            x = x + self.rbconvs[i](residual)
            x = F.relu(self.rbgrs[i](x))
            rbfeatures.append(x)

        logits_rb = F.interpolate(self.rbconvs[-1](x), size=(H, W))

        x = results[-1]
        for j, rttran in enumerate(self.rttrans):
            residual = results[3-j]
            x = rttran(x)

            diffH = residual.shape[2] - x.shape[2]
            diffW = residual.shape[3] - x.shape[3]
            x = F.pad(x, (diffW // 2, diffW - diffW//2,
                          diffH // 2, diffH - diffH//2))

            x = x + self.rtconvs[j](residual)
            x = F.relu(self.rtgrs[j](x))
            x = self.non_local_context(rbfeatures[j], x, j)
        
        logits_other = F.interpolate(self.last(x), size=(H, W))

        indices = list(range(44))
        # 21 heatmaps + 2 rooms -> background and outdoor
        indices = indices[:21+2] + [41] + indices[21+2:-1]
        # 21 heatmaps + 12 room classes + 1 icon -> no icon
        indices = indices[:21+12+1] + [42, 43] + indices[21+12+1:-2]

        cat = torch.cat((logits_other, logits_rb), dim=1)
        ordered = torch.index_select(cat, 1, torch.LongTensor(indices).to(self.device))

        ordered[:, :21] = torch.sigmoid(ordered[:, :21])

        return ordered


if __name__ == "__main__":

    with torch.no_grad():
        testin = torch.randn(1, 3, 1319, 619, device="cuda")
        model = DFPmodel()
        model.cuda()
        model.eval()
        ### Shared VGG encoder
        pred = model.forward(testin)
        # 0: 64x256x256, 1: 128x128x128
        # 2: 256x64x64, 3: 512x32x32, 4: 512x16x16
        print(pred.shape)