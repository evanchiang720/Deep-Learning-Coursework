import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        checkpoint = torch.load(load_ckpt_path, map_location='cpu')
        self.load_state_dict(checkpoint, strict=False)
        print(f"Successfully loaded transformer checkpoint from {load_ckpt_path}")

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        batch_size = x.shape[0]
        codebook_mapping, codebook_indices, q_loss = self.vqgan.encode(x)
        codebook_indices = codebook_indices.reshape(batch_size, -1)
        return codebook_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            # 第一種寫法
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: math.cos(math.pi * r / 2)
        elif mode == "square":
            # 第二種寫法
            def func(r):
                return 1 - r ** 2
            return func
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        z_indices=self.encode_to_z(x) #ground truth
        r = torch.rand(1).item()
        mask_rate = self.gamma(r)
        prob = torch.rand(z_indices.shape, device=z_indices.device)
        masked_indices = z_indices.clone() #numpy才是copy()
        masked_indices[prob <= mask_rate] = self.mask_token_id
        logits = self.transformer(masked_indices)  #transformer predict the probability of tokens
        
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, r, mask_num):
        logits = self.transformer(z_indices) # (b, n, codebook_size)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.softmax(logits, dim=2)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=2)

        ratio=r
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob) + 1e-10) + 1e-10)  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g # (b, n)
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        confidence[~mask_bc] = float('inf')
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values
        n = mask_bc.sum().item()
        mask_num = int(round(mask_num * self.gamma(ratio)))
        values, indices = torch.topk(confidence, k=mask_num, largest=False, dim=1)
        z_indices_predict[~mask_bc] = z_indices[~mask_bc]
        
        mask_bc = torch.zeros_like(mask_bc)
        mask_bc.scatter_(dim=1, index=indices, value=True)
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
