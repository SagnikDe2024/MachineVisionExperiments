from torch import nn
from torchinfo import summary
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

class EncoderVaeLayer1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv_norm = nn.BatchNorm2d(out_channels)
        self.conv_activation = nn.Mish()


class EncoderLayerVae(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )




if __name__ == '__main__':
    model = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)
    # model.features.Conv2dNormActivation = nn.Identity()
    model.avgpool = nn.Identity()
    model.classifier = nn.Identity()
    # for m in model.modules():
    #     print(f'{m.} , {m}')
    # model.load_state_dict(Inception_V3_QuantizedWeights)
    print(model)
    summary(model, input_size=(1,3, 299, 299))