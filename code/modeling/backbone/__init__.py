from modeling.backbone import resnet

def build_backbone(in_channels,backbone,output_stride, BatchNorm,Fusion = False):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm,pretrained=False)
    elif backbone== 'resnet50':
        return resnet.ResNet50(in_channels,output_stride, BatchNorm,pretrained=False,Fusion = Fusion)
    else:
        raise NotImplementedError