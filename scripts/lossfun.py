
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
from math import exp
# import kornia
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).double().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

class BCE_loss():
    def __init__(self):
        super().__init__()
    def cal(self,predictlabel, truelabel):
        validindex = torch.where(torch.sum(truelabel, axis=2) == 1)  
        criteria = nn.BCELoss()
        loss = criteria(predictlabel[validindex[0], validindex[1], :, validindex[2]],
                        truelabel[validindex[0], validindex[1], :, validindex[2]])
        return loss



class MSE_loss():
    def __init__(self):
        super().__init__()
    def cal(self,predictlabel, truelabel):
        validindex = torch.where(torch.sum(truelabel, axis=2) == 1)  
        valid_predictlabel=predictlabel[validindex[0], validindex[1], :, validindex[2]]
        valid_truelabel = truelabel[validindex[0], validindex[1], :, validindex[2]]
        label_index=torch.argmax(valid_truelabel,dim=1)/valid_truelabel.shape[1]
        predict_index=torch.argmax(valid_predictlabel,dim=1)/valid_truelabel.shape[1]
        criteria = nn.MSELoss()
        loss = criteria(label_index,predict_index)
        return loss

class BCE_MSE_loss():
    def __init__(self,balan_para):
        super().__init__()
        self.bp=balan_para
        self.BCEloss=BCE_loss()
        self.MSEloss=MSE_loss()
    def cal(self, predictlabel, truelabel):
        loss =self.BCEloss.cal(predictlabel,truelabel)+ self.bp*self.MSEloss.cal(predictlabel,truelabel)
        return loss

def CE_Loss(inputs, target, num_classes=0):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss( ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss
class Fusionloss(nn.Module):
    def __init__(self,coeff_int=1,coeff_grad=10,in_max=True, device='cuda'):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy(device=device)
        self.coeff_int=coeff_int
        self.coeff_grad=coeff_grad
        self.in_max=in_max
    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        if self.in_max:
            x_in_max=torch.max(image_y,image_ir)
        else:
            x_in_max=(image_y+image_ir)/2.0
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=self.coeff_int*loss_in+self.coeff_grad*loss_grad
        return loss_total

class Sobelxy(nn.Module):
    def __init__(self,device='cuda'):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):
    K, C, H, W = list(Ys.size())

    # compute statistics of the reference latent image Y
    muY_seq = F.conv2d(Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
        - muY_sq_seq
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # compute statistics of the test image X
    muX = F.conv2d(X, window, padding=ws // 2, groups=C).view(C, H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2,
                         groups=C).view(C, H, W) - muX_sq

    # compute correlation term
    sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
        - muX.expand_as(muY_seq) * muY_seq

    # compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    cs_map = torch.gather(cs_seq.view(K, -1), 0,
                          patch_index.view(1, -1)).view(C, H, W)
    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(- ((lY - 0.5) ** 2) /
                       denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())

    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()

    return q



class MEFSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return _mef_ssim(X, Ys, window, self.window_size,
                         self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)


class LpLssimLossweight(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        """
            Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        """
            Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)   # [window_size, 1]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # [1,1,window_size, window_size]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, image_in, image_out, weight):

        # Check if need to create the gaussian window
        (_, channel, _, _) = image_in.size()
        if channel == self.channel and self.window.data.type() == image_in.data.type():
            pass
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(image_out.get_device())
            window = window.type_as(image_in)
            self.window = window
            self.channel = channel

        # Lp
        Lp = torch.sqrt(torch.sum(torch.pow((image_in - image_out), 2))) 
        # Lp = torch.sum(torch.abs(image_in - image_out))  
        # Lssim
        Lssim = 1 - self._ssim(image_in, image_out, self.window, self.window_size, self.channel, self.size_average)
        return Lp + Lssim * weight, Lp, Lssim * weight
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        """
        inputs, targets = inputs.flatten(1), targets.flatten(1)
        inputs = inputs.sigmoid()
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks
        
class Loss_seg(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()

    def forward(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        loss_bce = self.bce_with_logits_loss(logit_mask, gt_mask.float())
        loss_dice = dice_loss(logit_mask, gt_mask, bsz)
        return loss_bce + loss_dice

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def tv_loss(x):

    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)

class TV_Loss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TV_Loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size=x.shape[0]
        return self.TVLoss_weight*tv_loss(x)/batch_size
###############################SSIM##################################################
def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret
def create_window1(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window1(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.cuda()
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.cuda()
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class Percep_Loss(nn.Module):
    def __init__(self):
        super(Percep_Loss, self).__init__()
        self.d = nn.MSELoss(size_average=True)
        vgg = vgg16(pretrained=True).cuda()
        # vgg.load_state_dict(torch.load(self.opt['vgg16_model']))
        # self.loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in self.loss_network.parameters():
        #     param.requires_grad = False

        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target,feature_layers=[0, 1, 2, 3],weights=[1,1,1,1]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for i,block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += weights[i] * self.d(x, y)
        return loss

class PixelLoss(nn.Module):
    """Loss function for the pixcel loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0):
        super(PixelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir, mask, *args, **kwargs):
        """Forward function.
        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): RGB image with shape (N, C, H, W).
        """
        pixel_max = torch.max(im_rgb, im_tir)
        mask_fusion = torch.where(mask>0, im_fusion, mask)
        mask_pixel = torch.where(mask>0, pixel_max.detach(), mask)

        pixel_mean = (im_rgb + im_tir)/2.0
        bg_mask = 1 - mask                                            
        bg_fusion = torch.where(bg_mask>0, im_fusion, bg_mask)
        bg_pixel = torch.where(bg_mask>0, pixel_mean.detach(), bg_mask) 

        mask_loss = self.L1_loss(mask_fusion, mask_pixel)
        bg_loss = self.L1_loss(bg_fusion, bg_pixel)
        pixel_loss = self.loss_weight * (mask_loss + bg_loss)

        return pixel_loss       
             
