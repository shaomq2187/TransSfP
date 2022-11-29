from modeling.backbone import resnet

def build_backbone(in_channels,backbone,output_stride, BatchNorm,Fusion = False):
    return resnet.ResNet50(in_channels,output_stride, BatchNorm,pretrained=False,Fusion = Fusion)
