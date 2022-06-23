from floortrans.models.hg_furukawa_original import *
from floortrans.models.dfp_adapted import *

def get_model(name, n_classes=None, version=None):
    if name == 'hg_furukawa_original':
        model = hg_furukawa_original(n_classes=n_classes)
        # model.init_weights()
    elif name == 'dfp':
        model = DFPmodel(pretrained=True, freeze=False, n_classes=n_classes)
    else:
        raise ValueError('Model {} not available'.format(name))

    return model


