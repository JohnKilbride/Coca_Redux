import torch
# import resnest
import torch.nn as nn
from torchsummary import summary
# from torchvision.models import AlexNet
from resnest.torch.models.resnet import *
from collections import OrderedDict

class HookedResNeSt50(nn.Module):
    '''
    Loads a "hooked" version of the ResNeSt50 classifier. The hooked version
    returns the feature maps at the end of each "block" in the architexture 
    (prior to downscaling)/ This is for training stuff like AutoEncoders
    '''
    def __init__(self, num_channels=3, *args):
        
        super().__init__(*args)
        
        self.num_channels = num_channels
        self.output_layers = [2,4,5,6,7]
        self.selected_out = OrderedDict()
        
        # Load in the ResNeSt50 Encoder
        self.model = ResNeSt50()
        self.fhooks = []
        
        # Configure input layer
        if num_channels != 3:
            self.model.conv1[0] = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Configure the hooks
        for i,l in enumerate(list(self.model._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.model,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        self.model(x)
        return self.selected_out

class HookedResNeSt101(nn.Module):
    '''
    Loads a "hooked" version of the ResNeSt101 classifier. The hooked version
    returns the feature maps at the end of each "block" in the architexture 
    (prior to downscaling)/ This is for training stuff like AutoEncoders
    '''
    def __init__(self, num_channels=3, *args):
        
        super().__init__(*args)
        
        self.output_layers = [2,4,5,6,7]
        self.selected_out = OrderedDict()
        
        # Load in the ResNeSt50 Encoder
        self.model = ResNeSt101()
        self.fhooks = []

        # Configure input layer
        if num_channels != 3:
            self.model.conv1[0] = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Configure the hooks
        for i,l in enumerate(list(self.model._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.model,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        self.model(x)
        return self.selected_out

class HookedResNeSt200(nn.Module):
    '''
    Loads a "hooked" version of the ResNeSt200 classifier. The hooked version
    returns the feature maps at the end of each "block" in the architexture 
    (prior to downscaling)/ This is for training stuff like AutoEncoders
    '''
    def __init__(self, num_channels=3, *args):
        
        super().__init__(*args)
        
        self.output_layers = [2,4,5,6,7]
        self.selected_out = OrderedDict()
        
        # Load in the ResNeSt50 Encoder
        self.model = ResNeSt200()
        self.fhooks = []
        
        print(self.model.conv1)
        
        # Configure input layer
        if num_channels != 3:
            self.model.conv1[0] = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Configure the hooks
        for i,l in enumerate(list(self.model._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.model,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        self.model(x)
        return self.selected_out


class ResNeSt50(ResNet):
    """
    Loads ResNeSt50 without the classification head or flattening layers.
    """
    # pylint: disable=unused-variable
    def __init__(self):
        super(ResNeSt50, self).__init__(
            Bottleneck, [3, 4, 6, 3], radix=2, groups=1,  bottleneck_width=64, 
            deep_stem=True, stem_width=32, avg_down=True,avd=True, avd_first=False
            )
        self.avgpool = Identity()
        self.fc = Identity()
        return 
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
class ResNeSt101(ResNet):
    """
    Loads ResNeSt101 without the classification head or flattening layers.
    """
    # pylint: disable=unused-variable
    def __init__(self):
        super(ResNeSt101, self).__init__(
            Bottleneck, [3, 4, 23, 3], radix=2, groups=1, bottleneck_width=64,
            deep_stem=True, stem_width=64, avg_down=True, avd=True, avd_first=False
            )
        self.avgpool = Identity()
        self.fc = Identity()
        return 
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
class ResNeSt200(ResNet):
    """
    Loads ResNeSt101 without the classification head or flattening layers.
    """
    # pylint: disable=unused-variable
    def __init__(self):
        super(ResNeSt200, self).__init__(
            Bottleneck, [3, 24, 36, 3], radix=2, groups=1, bottleneck_width=64,
            deep_stem=True, stem_width=64, avg_down=True, avd=True, avd_first=False
            )
        self.avgpool = Identity()
        self.fc = Identity()
        return 
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

class Identity(nn.Module):
    '''
    Used to delete layers of a neural network.
    
    Source: https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/2
    '''
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

if __name__ == "__main__":
    
    # Number of input channels
    num_channels = 4
    
    # Set the batch size
    batch_size = 8
    
    # Load in each of the hooked models
    model1 = HookedResNeSt50(num_channels)
    model2 = HookedResNeSt101(num_channels)
    model3 = HookedResNeSt200(num_channels)
    
    # Pass a batch through the model
    input = torch.rand(batch_size, num_channels, 256, 256).to('cpu')
    output1 = model1(input)
    output2 = model2(input)
    output3 = model3(input)
    
    # Check the size of the outputs from the lowest layer
    print('model 1 shape', output1['layer4'].shape)
    print('model 2 shape', output2['layer4'].shape)
    print('model 3 shape', output3['layer4'].shape)
