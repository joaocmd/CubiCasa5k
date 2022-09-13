from floortrans.models.hg_furukawa_original import hg_furukawa_original
from floortrans.models import dfp_conv
from floortrans.models import dfp_resnet34
from floortrans.models import dfp_resnet34_encoder_conv
from floortrans.models import dfp_resnet50
from floortrans.models import dfp_resnet50_encoder_conv
from floortrans.models import deeplabv3

def get_model(name, n_classes=None, version=None, device='cpu'):
    last_conv = 'last-conv' in name
    if name == 'hg_furukawa_original':
        model = hg_furukawa_original(n_classes=n_classes, device=device)
        # model.init_weights()
    elif name in ('dfp','dfp-last-conv'):
        model = dfp_conv.Model(pretrained=True, freeze=False, n_classes=n_classes, last_conv=last_conv, device=device)
    elif name in ('dfp-resnet34', 'dfp-resnet34-last-conv'):
        model = dfp_resnet34.Model(pretrained=True, freeze=False, n_classes=n_classes, last_conv=last_conv, device=device)
    elif name in ('dfp-resnet34-encoder-conv', 'dfp-resnet34-encoder-conv-last-conv'):
        model = dfp_resnet34_encoder_conv.Model(pretrained=True, freeze=False, n_classes=n_classes, last_conv=last_conv, device=device)
    elif name in ('dfp-resnet50', 'dfp-resnet50-last-conv'):
        model = dfp_resnet50.Model(pretrained=True, freeze=False, n_classes=n_classes, last_conv=last_conv, device=device)
    elif name in ('dfp-resnet50-encoder-conv', 'dfp-resnet50-encoder-conv-last-conv'):
        model = dfp_resnet50_encoder_conv.Model(pretrained=True, freeze=False, n_classes=n_classes, last_conv=last_conv, device=device)
    elif name == 'deeplabv3':
        model = deeplabv3.Model(pretrained=True, n_classes=n_classes, device=device)
    else:
        raise ValueError('Model {} not available'.format(name))

    return model
