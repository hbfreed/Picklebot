import torch
import torch.nn as nn
import torch.nn.init as init


def calculate_accuracy(outputs,labels):
    predicted_classes = torch.argmax(outputs,dim=1).to(labels.device)
    num_correct = torch.sum(predicted_classes == labels).item()
    return num_correct

def initialize_mobilenetv3_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
            # Initialize convolutional layers with appropriate initialization
            if hasattr(module, 'nonlinearity') and module.nonlinearity == 'hardswish':
                init.xavier_uniform_(module.weight, gain=init.calculate_gain('leaky_relu'))
            else:
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            # Initialize Batch Normalization layers with small values and biases to zero
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            # Initialize fully connected (linear) layers with appropriate initialization
            if hasattr(module, 'nonlinearity') and module.nonlinearity == 'hardswish':
                init.xavier_uniform_(module.weight, gain=init.calculate_gain('leaky_relu'))
            else:
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')

