from torchvision import models
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DFPResNet34ConvModel(torch.nn.Module):
    """Model of the Deep FloorPlan Recognition [1].

    Receives images as inputs and outputs the pixel-wise classification concerning
    the different room regions and boundaries.  This code is based on the Pytorch
    implementation for the Deep Floor Plan paper [2].

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
        The number of classes to consider.

    References
    ----------
    1 - https://arxiv.org/pdf/1908.11025.pdf
    2 - https://github.com/zcemycl/PyTorch-DeepFloorplan/blob/main/net.py
    """
    def __init__(self, pretrained: bool=True, freeze: bool=True, n_classes: int=44, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.n_classes = n_classes
        # ----------------------------------------------------
        # 1. Initialize VGG encoder (component 1)
        # ----------------------------------------------------
        # Output will be shared with RB and RT
        self._initializeVGG(pretrained, freeze)

        # ----------------------------------------------------
        # 2. Room Boundary Prediction 
        # ----------------------------------------------------
        # We will have a skip-connection like model. We use 
        # VGG16 encoder's shared features and pass it through
        # **upsampling** (rbtrans) layer. We add element-wise
        # the upsampled representation with the encoder's
        # representation of the previous layer (after applying
        # a convolution with rbconvs). Finally, we apply a conv
        # (rbgrs) to obtain the final representation.
        # ----------------------------------------------------
        # List of channel dimensions per layer 
        rblist = [512, 256, 128, 64, 64, 3]
        # ^Note: Last layer size equals number of RB classes: walls, doors, windows

        # *rbtrans*: VGG decoder that will be used to output the classes
        self.rbtrans = nn.ModuleList([self._transconv2d(
            rblist[i], rblist[i+1], 4, 2, 1) for i in range(len(rblist)-2)])
        
        # *rbconvs* and *rbgrs*: 
        self.rbconvs = nn.ModuleList([self._conv2d(
            rblist[i], rblist[i+1], 3, 1, 1) for i in range(len(rblist)-1)])
        
        self.rbgrs = nn.ModuleList([self._conv2d(
            rblist[i], rblist[i], 3, 1, 1) for i in range(1, len(rblist)-1)])
        self.rblastconv = self._conv2d(3, 3, 1)

        # ----------------------------------------------------
        # 3. Room Type Prediction 
        # ----------------------------------------------------
        rtlist = [512, 256, 128, 64, 64]

        # *rttrans*: VGG decoder 
        self.rttrans = nn.ModuleList([self._transconv2d(
            rtlist[i], rtlist[i+1], 4, 2, 1) for i in range(len(rtlist)-1)])
        
        # *rtconvs*: convolution used 
        self.rtconvs = nn.ModuleList([self._conv2d(
            rtlist[i], rtlist[i+1], 3, 1, 1) for i in range(len(rtlist)-1)])
        self.rtgrs = nn.ModuleList([self._conv2d(
            rtlist[i], rtlist[i], 3, 1, 1) for i in range(1, len(rtlist))])
        self.rtlastconv = self._conv2d(41, 41, 1)

        # ----------------------------------------------------
        # 4. Attention Mechanism 
        # ----------------------------------------------------
        # Attention Non-local context
        clist = [256, 128, 64, 64]
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

        # Last layer
        # Question: Why n_classes - 3?
        # Since the RB outputs 3 classes
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
        # encmodel = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        encmodel = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        # encmodel = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        if freeze:
            for child in encmodel.children():
                for param in child.parameters():
                    param.requires_grad = False

        features = list(encmodel.children())[:-2]
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
    
    def non_local_context(self, t1, t2, idx, stride=4):
        """Apply the spatial contextual attention module"""
        # Example of invocations: rbfeatures[j], x, j
        # t1 is the RB features
        # t2 is the current RT feature 
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
        # VGG16 Encoder layers where  layer size shrinks
        DOWNSAMPLE_LAYERS = {0, 3, 5, 6, 7}

        # 21 heatmap classes
        # 12 room classes
        # room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
        # 11 icon classes
        # icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

        # The input fed into the network is of size N x C x H x W, where
        # - N is the batch size
        # - C is the number of channels
        # - H is the picture height
        # - W is the picture width
        # In this case, we're processing RGB pictures 256x256, so 
        # the size is [N, 3, 256, 256]
        N, C, H, W = x.shape

        # ------------------------------------------------
        # Encoder forward
        # ------------------------------------------------
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in DOWNSAMPLE_LAYERS:
                results.append(x)
        # ^Note: results[-1] will have the "shared features"
        # from the VGG encoder. It's size is N x 512 x 8 x 8
        
        # ------------------------------------------------
        # Room Boundary
        # ------------------------------------------------
        # rbtrans contains 4 layers: 
        # (512, 256), (256, 128), (128, 64), (64, 32)
        rbfeatures = []
        last = len(results) - 2
        for i, rbtran in enumerate(self.rbtrans):
            residual = results[last-i]
            x = rbtran(x)
            # Update: @joaocmd
            # Check if we need to pad due to changes in the images size
            # ---------------------------------------------------------------
            # Question: Why do we perform padding at this stage and not 
            # earlier in the process before feeding images to the network? 
            # Answer (2022-Aug-08): for flexibility. By resizing images later 
            # in the network we support images of any size to be fed to the
            # network. Doing it earlier might lead to loss of information, as
            # upsampling the images might make them blurred or too small.
            # Another alternative could be to crop several images and combine
            # them in the end.
            # --------------------------------------------------------------- 
            diffH = residual.shape[2] - x.shape[2]
            diffW = residual.shape[3] - x.shape[3]
            if diffH != 0 or diffW != 0:
                pad_horizont = diffW // 2
                pad_vertical = diffH // 2
                # (pad_left, pad_right, pad_top, pad_bottom)
                padding = (pad_horizont, pad_horizont + diffW % 2,
                           pad_vertical, pad_vertical + diffH % 2)
                x = F.pad(x, padding)
            # end update @joaocmd ------------------------------------------
            x = x + residual
            x = F.relu(self.rbgrs[i](x))
            rbfeatures.append(x)
        # x.shape: batch_size x 32 x 128 x 128, 
        # rbconvs[-1] is a conv layer from in=32 to out=3 channels
        rb_outputs = self.rbconvs[-1](x)
        # Resize to H x W
        logits_rb = F.interpolate(rb_outputs, size=(H, W))
        logits_rb = self.rblastconv(logits_rb)
        
        # ------------------------------------------------
        # Room Type prediction
        # ------------------------------------------------
        # Similarly to the RB network it applies a skip
        # connection (using a residual from the VGG encoder)
        # and the transformation of the shared feature 
        # representation.
        # ------------------------------------------------ 
        x = results[-1] # size: N x 256 x 8 x 8
        for j, rttran in enumerate(self.rttrans):
            residual = results[last-j]
            x = rttran(x)
            # Update: @joaocmd
            diffH = residual.shape[2] - x.shape[2]
            diffW = residual.shape[3] - x.shape[3]
            if diffH != 0 or diffW != 0:
                pad_horizont = diffW // 2
                pad_vertical = diffH // 2
                # (pad_left, pad_right, pad_top, pad_bottom)
                padding = (pad_horizont, pad_horizont + diffW % 2,
                           pad_vertical, pad_vertical + diffH % 2)
                x = F.pad(x, padding)
            # end update @joaocmd ------------------------------------------
            x = x + residual
            x = F.relu(self.rtgrs[j](x))
            x = self.non_local_context(rbfeatures[j], x, j)
        
        logits_other = F.interpolate(self.last(x), size=(H, W)) # 16 x 41 x 128 x 128 --> 16 x 41 x 256 x 256
        logits_other = self.rtlastconv(logits_other)
        # Update: @joao ----------------------------------------------------
        # We have three tasks in total: 1x regression + 2 multi-class
        # 1. **Regression tasks** (21) for the junctions heatmaps. These are
        # passed through a sigmoid and will be alocated to the first 21 
        # positions of the output. 
        # 2. **Classification** (12): rooms classification.
        # 3. **Classification** (11): icons classification.
        # 
        # The output of logits_other predicts 41 classes, which we will combine
        # 
        indices = list(range(self.n_classes))
        # 21 heatmaps + 2 rooms -> background and outdoor
        indices = indices[:21+2] + [41] + indices[21+2:-1]
        # 21 heatmaps + 12 room classes + 1 icon -> no icon
        indices = indices[:21+12+1] + [42, 43] + indices[21+12+1:-2]

        cat = torch.cat((logits_other, logits_rb), dim=1)
        ordered = torch.index_select(cat, 1, torch.LongTensor(indices).to(self.device))

        ordered[:, :21] = torch.sigmoid(ordered[:, :21])

        # Update @joao: original code was returning two different outputs
        return ordered

if __name__ == "__main__":

    with torch.no_grad():
        testin = torch.randn(1, 3, 512, 512, device="cuda")
        # testin = torch.randn(1, 3, 1213, 956, device="cuda")
        model = DFPResNet34ConvModel()
        model.cuda()
        model.eval()
        ### Shared VGG encoder
        pred = model(testin)
        # 0: 64x256x256, 1: 128x128x128
        # 2: 256x64x64, 3: 512x32x32, 4: 512x16x16
        print(pred.shape)
