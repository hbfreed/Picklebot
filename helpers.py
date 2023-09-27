import torch


def calculate_accuracy(outputs,labels):
    predicted_classes = torch.argmax(outputs,dim=1).to(labels.device)
    num_correct = torch.sum(predicted_classes == labels).item()
    return num_correct