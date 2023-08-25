from Pyramid_Swin_Transformer import Pyramidswin_r
import torch.nn as nn

def get_model(num_classes=1000):
    return Pyramidswin_r(num_classes=num_classes).cuda()

def get_loss():
    return nn.CrossEntropyLoss().cuda()
