from torch.nn.functional import mse_loss
import torch

def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)

def completion_network_loss_P(background, person, output, lefttop, height = 64 , width = 32, devices = 'cuda:0'):
    """
    * background:
        - shape: 3 * 256 * 256
    """
    mask = torch.zeros(size = output.shape).to(devices)
    batchsize = output.shape[0]
    for i in range(batchsize):
        left , top = lefttop[i][0],lefttop[i][1]
        mask[ i , top : top + height , left : left + width,:]  = 1

    return 0.7*mse_loss(output * mask, person * mask) + 0.3*mse_loss(output, background)

