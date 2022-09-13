from floortrans.models.hg_furukawa_original import hg_furukawa_original
from floortrans.models.dfp_conv import DFPConvModel
from floortrans.models.dfp_resnet50 import DFPResNet50Model
from floortrans.models.dfp_resnet34 import DFPResNet34Model
from floortrans.models import dfp_resnet34_encoder_conv
from floortrans.models.deeplabv3 import DeepLabModel

def get_model(name, n_classes=None, version=None, device='cpu'):
    last_conv = 'last-conv' in name
    if name == 'hg_furukawa_original':
        model = hg_furukawa_original(n_classes=n_classes, device=device)
        # model.init_weights()
    elif name in ('dfp','dfp-last-conv'):
        model = DFPConvModel(pretrained=True, freeze=False, n_classes=n_classes, last_conv=last_conv, device=device)
    elif name in ('dfp-resnet34', 'dfp-resnet34-last-conv'):
        model = DFPResNet34Model(pretrained=True, freeze=False, n_classes=n_classes, last_conv=last_conv, device=device)
    elif name in ('dfp-resnet34-encoder-conv'):
        model = dfp_resnet34_encoder_conv.Model(pretrained=True, freeze=False, n_classes=n_classes, last_conv=last_conv, device=device)
    elif name in ('dfp-resnet50', 'dfp-resnet50-last-conv'):
        model = DFPResNet50Model(pretrained=True, freeze=False, n_classes=n_classes, last_conv=last_conv, device=device)
    elif name == 'deeplabv3':
        model = DeepLabModel(pretrained=True, n_classes=n_classes, device=device)
    else:
        raise ValueError('Model {} not available'.format(name))

    return model
