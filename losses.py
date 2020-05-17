from torch.nn.functional import mse_loss
import torch

def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)

def completion_network_loss_P(background, person, mask, output, lefttop, height = 64 , width = 32, devices = 'cuda:0'):
    """
    * background:
        - shape: batchsize * 3 * 256 * 256
    * output:
        - shape: batchsize * 3 * 256 * 256
    """
    output_crop = torch.zeros([output.shape[0],3,128,64]).to(devices)
    person_crop = torch.zeros([output.shape[0],3,128,64]).to(devices)
    mask_crop = torch.zeros([output.shape[0],3,128,64]).to(devices)
    
    #mask = torch.zeros(size = output.shape).to(devices)
    batchsize = output.shape[0]
    for i in range(batchsize):
        left , top = lefttop[i][0],lefttop[i][1]
        output_crop[i,:,:,:] = output[i,:,top : top + height , left : left + width]
        person_crop[i,:,:,:] = person[i,:,top : top + height , left : left + width]
        mask_crop[i,:,:,:] = mask[i,:,top : top + height , left : left + width]
        #mask[ i ,:, top : top + height , left : left + width]  = 1

    return 0.5*mse_loss(output_crop, person_crop) + 0.35*mse_loss(output, background) + 0.15*(-1)*mse_loss(mask_crop,person_crop)

