from __future__ import absolute_import

from networks.AugmentCE2P import resnet101_1out, resnet101_3out

__factory = {
    'resnet101_1': resnet101_1out,
    'resnet101_3': resnet101_3out,
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)