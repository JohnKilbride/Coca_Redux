import torch
import torch.nn as nn
import random
import torchvision.transforms as transforms

def load_augmentations(use_noise=True):
    '''
    Formats all of the different transforms for random augmentation during semantic segmentation.
    The augmentations here consist of simple rotations and flips.

    Returns
    -------
    data_transforms : torch transform
      
    '''
    # Define the transforms that need to be use
    if use_noise:
        print("\n-----\nUsing Noise\n-----\n")
        augmentations = transforms.Compose([
            transforms.RandomChoice([
                NullTransform(),                          # No transformation
                RotateRight2DTensor90(),                # 90
                RotateRight2DTensor180(),               # 180
                RotateRight2DTensor270(),               # 270
            ]),
            transforms.RandomChoice([
                NullTransform(),                          # No transformation
                HorizontaFlip2DTensor()                   # Rotation
            ]),
            transforms.RandomChoice([
                NullTransform(),                # No transformation
                NullTransform(),                # No transformation
                RandomPixeldrop(),              # Randomly mask pixels
            ])
        ])        

    else:
        print("\n-----\nNot using Noise\n-----\n")
        augmentations = transforms.Compose([
            transforms.RandomChoice([
                NullTransform(),                          # No transformation
                RotateRight2DTensor90(),                # 90
                RotateRight2DTensor180(),               # 180
                RotateRight2DTensor270(),               # 270
            ]),
            transforms.RandomChoice([
                NullTransform(),                          # No transformation
                HorizontaFlip2DTensor()                   # Rotation
            ])
        ])
    
    return augmentations

class RandomPixeldrop(nn.Module):

    def __init__(self, mask_p=None):
        return None
        
    def __call__(self, paired_input):
        
        # Unpack the inputs
        features, label = paired_input
        
        # Get the shape of the input
        f_shp = features.shape
        
        # If on CPU then you don't need to worry about the device
        # if features.get_device() == -1:
        mask_tensor = torch.rand((f_shp[2],f_shp[-1])).gt(torch.rand(1) * 0.25)
        return (features * mask_tensor, label)
    
    def __repr__(self):
        return self.__class__.__name__

class NullTransform(object):
    """
    Placeholder transformation. Does not change the input at all.
    """
    def __init__(self):
        return

    def __call__(self, paired_input):

        features, label = paired_input
        return (features, label)
     
    def __repr__(self):
        return self.__class__.__name__

class RotateRight2DTensor90(object):

    def __init__(self):
        return None

    def __call__(self, paired_input):
        features, label = paired_input
        return (torch.rot90(features, 1, [-2,-1]), torch.rot90(label, 1, [-2,-1]).contiguous()) 
 
    def __repr__(self):
        return self.__class__.__name__
    
class RotateRight2DTensor180(object):
    
    def __init__(self):
        return None

    def __call__(self, paired_input):
        features, label = paired_input
        return (torch.rot90(features, 2, [-2,-1]), torch.rot90(label, 2, [-2,-1]).contiguous()) 
 
    def __repr__(self):
        return self.__class__.__name__
    
class RotateRight2DTensor270(object):
    
    def __init__(self):
        return None

    def __call__(self, paired_input):
        features, label = paired_input
        return (torch.rot90(features, 3, [-2,-1]), torch.rot90(label, 3, [-2,-1]).contiguous()) 
 
    def __repr__(self):
        return self.__class__.__name__

class HorizontaFlip2DTensor(object):

    def __init__(self):
        return None

    def __call__(self, paired_input):

        features, label = paired_input
        return (torch.transpose(features, -2, -1), torch.transpose(label, -2, -1).contiguous())

    def __repr__(self):
        return self.__class__.__name__
    
    