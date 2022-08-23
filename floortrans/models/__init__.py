from floortrans.models.hg_furukawa_original import *
from floortrans.models.dfp_adapted import DFPmodel
from floortrans.models.dfp_conv import DFPConvModel
from floortrans.models.dfp_resnet50 import DFPResNet50Model
from floortrans.models.dfp_resnet50_conv import DFPResNet50ConvModel
from floortrans.models.dfp_resnet34 import DFPResNet34Model
from floortrans.models.dfp_resnet34_conv import DFPResNet34ConvModel

def get_model(name, n_classes=None, version=None):
    if name == 'hg_furukawa_original':
        model = hg_furukawa_original(n_classes=n_classes)
        # model.init_weights()
    elif name == 'dfp':
        model = DFPmodel(pretrained=True, freeze=False, n_classes=n_classes)
    elif name == 'dfp-last-conv':
        model = DFPConvModel(pretrained=True, freeze=False, n_classes=n_classes)
    elif name == 'dfp-resnet34':
        model = DFPResNet34Model(pretrained=True, freeze=False, n_classes=n_classes)
    elif name == 'dfp-resnet34-conv':
        model = DFPResNet34ConvModel(pretrained=True, freeze=False, n_classes=n_classes)
    elif name == 'dfp-resnet50':
        model = DFPResNet50Model(pretrained=True, freeze=False, n_classes=n_classes)
    elif name == 'dfp-resnet50-conv':
        model = DFPResNet50ConvModel(pretrained=True, freeze=False, n_classes=n_classes)
    else:
        raise ValueError('Model {} not available'.format(name))

    return model


