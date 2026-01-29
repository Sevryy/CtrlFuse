from segment_anything import sam_model_registry
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import numbers
from einops import rearrange
from PIL import Image, ImageDraw, ImageFilter
from model.base.transformer_decoder import transformer_decoder
from segment_anything.modeling.transformer import Attention
from typing import Any, Optional, Tuple, Type
##################################################################################################
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 128, scale: Optional[float] = None) -> None:##num_pos_feats=C/2
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
        

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2,inplace=True)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2,inplace=True)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1,inplace=True)

class HighResCrossAttention(nn.Module):
    def __init__(self, dim=256, heads=8, downsample_ratio=2):
        super().__init__()
        # downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=downsample_ratio, stride=downsample_ratio),
            nn.GroupNorm(8, dim)
        )
        
        # upsample
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=downsample_ratio, mode='bilinear'),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim)
        )
        
        # attention block
        self.attn = nn.MultiheadAttention(dim, heads)
        
        #output
        self.out_proj = nn.Conv2d(dim*2, dim, kernel_size=1)

    def forward(self, query, spatial_feat):
        """ 
        input:
            query: [B,50,256] 
            spatial_feat: [B,256,600,800] 
        output:
            [B,256,600,800] 
        """
        B, C, H, W = spatial_feat.shape
        
        # 1. downsample Key/Value [B,256,600,800] -> [B,256,300,400]
        kv_down = self.downsample(spatial_feat)
        
        # 2. flatten [B,256,30,40] -> [B,1200,256]
        kv_seq = kv_down.flatten(2).transpose(1, 2)
        
        # 3.cross attetion [B,50,256] -> [B,50,256]
        attn_out, _ = self.attn(
            query.transpose(0, 1),  # [50,B,256]
            kv_seq.transpose(0, 1), # [1200,B,256]
            kv_seq.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)  # [B,50,256]
        
        # 4. Broadcast the attention results back to the spatial dimensions.
        # Method: Expand to [B, 256, 30, 40] via 1x1 convolution, then perform upsampling.
        attn_spatial = attn_out.transpose(1, 2).view(B, C, 5, 10)  
        attn_spatial = F.interpolate(attn_spatial, size=(int(H/2), int(W/2)), mode='nearest')
        
        # 5. Upsample to the original size. [B,256,30,40] -> [B,256,600,800]
        attn_highres = self.upsample(attn_spatial)
        #attn_highres = F.interpolate(attn_spatial, size=(H, W), mode='bilinear', align_corners=True) 
        # 6. Fuse with the original features (residual connection)
        #print(f"spatial_feat shape: {spatial_feat.shape}")
        #print(f"attn_highres shape: {attn_highres.shape}")
        output = self.out_proj(torch.cat([spatial_feat, attn_highres], dim=1))
        return output  # [B,256,600,800]
##########################################################################################################
class  net(nn.Module):
    def __init__(self,
        use_original_imgsize = False,
        decoder_num_heads=8,
        num_query = 50,
        dim=256,
        ffn_factor=4,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias',
        out_channel=1,):
        super().__init__()
        self.sam_model = sam_model_registry['vit_h']('/root/autodl-tmp/CtrlFuse/pth/sam_vit_h_4b8939.pth')
        self.sam_model.eval()
      
        self.merge_v = nn.Sequential(
            nn.Conv2d(dim*2+1, dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )    
        self.merge_i = nn.Sequential(
            nn.Conv2d(dim*2+1, dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )       
        self.transformer_decoder_v = transformer_decoder(num_query, dim, dim*2)
        self.transformer_decoder_i = transformer_decoder(num_query, dim, dim*2)
        
        self.token2image1 = HighResCrossAttention()
        self.token2image2 = HighResCrossAttention()

        self.feature_extractor1 = nn.Sequential(
            ConvLeakyRelu2d(3,16),
            RGBD(16,32),
            RGBD(32,64),
            ConvLeakyRelu2d(64,128),
            ConvLeakyRelu2d(128,256)
        )
        self.feature_extractor2 = nn.Sequential(
            ConvLeakyRelu2d(3,16),
            RGBD(16,32),
            RGBD(32,64),
            ConvLeakyRelu2d(64,128),
            ConvLeakyRelu2d(128,256)
        )      
        self.image_decoder = nn.Sequential(
            ConvBnLeakyRelu2d(256,128),
            ConvBnLeakyRelu2d(128,64),
            ConvBnLeakyRelu2d(64,32),
            ConvBnLeakyRelu2d(32,16),
            ConvBnTanh2d(16, 1)
        )
        self.image_decoder_iv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.merge_2c = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )      

    def forward_img_encoder(self, query_img):
        query_img = F.interpolate(query_img, (1024,1024), mode='bilinear', align_corners=True)
        with torch.no_grad():
            query_feats= self.sam_model.image_encoder(query_img)
        return  query_feats

    def get_pormpt(self, protos, points_mask=None):
        if points_mask is not None :
            point_mask = points_mask

            postivate_pos = (point_mask.squeeze(0).nonzero().unsqueeze(0) + 0.5) * 64 -0.5
            postivate_pos = postivate_pos[:,:,[1,0]]
            point_label = torch.ones(postivate_pos.shape[0], postivate_pos.shape[1]).to(postivate_pos.device)
            point_prompt = (postivate_pos, point_label)
        else:
            point_prompt = None
        protos = protos
        return  protos, point_prompt

    def forward_prompt_encoder(self, points=None, boxes=None, protos=None, masks=None):
        q_sparse_em, q_dense_em = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                protos=protos,
                masks=None)
        return  q_sparse_em, q_dense_em
    
    def forward_mask_decoder(self, query_feats, q_sparse_em, q_dense_em, ori_size=(512,512)):
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=query_feats,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=q_sparse_em,
                dense_prompt_embeddings=q_dense_em,
                multimask_output=False)
        low_masks = F.interpolate(low_res_masks, size=ori_size, mode='bilinear', align_corners=True)

        # binary_mask = normalize(threshold(low_masks, 0.0, 0))
        binary_mask = torch.where(low_masks > 0, 1, 0)
        return low_masks, binary_mask

    def get_pseudo_mask(self, tmp_supp_feat, query_feat_4, mask):
        resize_size = tmp_supp_feat.size(2)
        tmp_mask = F.interpolate(mask, size=(resize_size, resize_size), mode='bilinear', align_corners=True)

        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
        q = query_feat_4
        s = tmp_supp_feat_4
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s               
        tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
        tmp_supp = tmp_supp.permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
        return corr_query

    def mask_feature(self, features, support_mask):
        mask = support_mask
        supp_feat = features * mask

        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def forward(self, condition, vis_img, ir_img, support_mask, name, points_mask=None):
        if condition == 'mask':
            support_mask_ori = support_mask
        #with torch.no_grad():
            #-------------save_sam_img_feat-------------------------
        #    query_samfeat = self.get_feat_from_np(query_img, name)    
        B,C, h, w = vis_img.shape  
        #Generate the preliminary fused image.
        vis_feat = self.feature_extractor1(vis_img)#b,256,h,w  #Enc
        ir_feat = self.feature_extractor2(ir_img)#b,256,h,w    #Enc
        Iref_feat = self.merge_2c(torch.cat([vis_feat,ir_feat],1))#b,256,h,w   #obtain Fref
        I_ref = self.image_decoder(Iref_feat)#b,1,h,w
        ################################################RP Encoder####################################################
        Iref_feat1 = F.interpolate(Iref_feat.float(), size=(64,64), mode='nearest')#bsz,256,64,64
        vis_feat1 = F.interpolate(vis_feat.float(), size=(64,64), mode='nearest')#bsz,256,64,64
        ir_feat1 = F.interpolate(ir_feat.float(), size=(64,64), mode='nearest')#bsz,256,64,64
        support_mask = F.interpolate(support_mask_ori.float(), size=(64,64), mode='nearest')#bsz,1,64,64

        prototype_v = self.mask_feature(vis_feat1, support_mask)#b,256,1,1
        prototype_i = self.mask_feature(ir_feat1, support_mask)#b,256,1,1

        supp_feat_bin_v = prototype_v.repeat(1, 1, Iref_feat1.shape[2], Iref_feat1.shape[3])
        supp_feat_bin_i = prototype_i.repeat(1, 1, Iref_feat1.shape[2], Iref_feat1.shape[3])

        pseudo_mask_v = self.get_pseudo_mask(vis_feat1, Iref_feat1, support_mask)#b,1,64,64
        pseudo_mask_i = self.get_pseudo_mask(ir_feat1, Iref_feat1, support_mask)#b,1,64,64

        supp_feat_v = self.merge_v(torch.cat([vis_feat1, supp_feat_bin_v, support_mask*10], 1))
        supp_feat_i = self.merge_i(torch.cat([ir_feat1, supp_feat_bin_i, support_mask*10], 1))
        query_feat_v = self.merge_v(torch.cat([Iref_feat1, supp_feat_bin_v, pseudo_mask_v*10], 1))
        query_feat_i = self.merge_i(torch.cat([Iref_feat1, supp_feat_bin_i, pseudo_mask_i*10], 1))

        protos_v = self.transformer_decoder_v(query_feat_v ,supp_feat_v)#b,50,256
        protos_vis, point_prompt_vis = self.get_pormpt(protos_v, points_mask)

        protos_i = self.transformer_decoder_i(query_feat_i,supp_feat_i)#b,50,256
        protos_ir, point_prompt_ir = self.get_pormpt(protos_i, points_mask)

        q_sparse_em_v, q_dense_em_v = self.forward_prompt_encoder(
                points=point_prompt_vis,
                boxes=None,
                protos=protos_vis,
                masks=None)  
        q_sparse_em_i, q_dense_em_i = self.forward_prompt_encoder(
                points=point_prompt_ir,
                boxes=None,
                protos=protos_ir,
                masks=None)   
        #######################################################################################################

        I_ref_samf = self.forward_img_encoder(I_ref.repeat(1, 3, 1, 1))

        logit_mask_v, binary_mask_v = self.forward_mask_decoder(I_ref_samf, q_sparse_em_v, q_dense_em_v, ori_size=(h, w))
        logit_mask_i, binary_mask_i = self.forward_mask_decoder(I_ref_samf, q_sparse_em_i, q_dense_em_i, ori_size=(h, w))
       
        combined_logit = torch.maximum(logit_mask_v, logit_mask_i)  
        pred_mask_v = torch.sigmoid(logit_mask_v) > 0.5  # perform binarization
        pred_mask_i = torch.sigmoid(logit_mask_i) > 0.5  # perform binarization
        pred_mask = torch.sigmoid(combined_logit) > 0.5  # perform binarization
        
        ######################################Prompt-Semantic Fusion Module##########################################
        q_vis = self.token2image1(q_sparse_em_v,vis_feat)
        q_ir = self.token2image2(q_sparse_em_i,ir_feat)
        ##############################################################################################################
        q = q_vis*pred_mask_v*1 + q_ir*pred_mask_i*1 + Iref_feat ########## *5 / *10 
        fusionfeature = self.image_decoder(q) #Dec
      
        return I_ref,logit_mask_v,logit_mask_i,pred_mask_v,pred_mask_i,pred_mask,fusionfeature

