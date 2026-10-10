import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse + 1e-9)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.current_epoch = current_epoch
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.num_epoch = args.num_epoch
        self.steps_per_epoch = 0
        self.current_iter = 0
        self.total_iter = 0
        
    def update(self):
        # TODO
        self.current_iter += 1
    
    def get_beta(self):
        # TODO
        if self.kl_anneal_type == 'Cyclical':
            return self.frange_cycle_linear(self.total_iter, 0.0, 1.0, self.kl_anneal_cycle, self.kl_anneal_ratio)
        elif self.kl_anneal_type == 'Monotonic':
            return self.frange_cycle_linear(self.total_iter, 0.0, 1.0, 1, self.kl_anneal_ratio)
        elif self.kl_anneal_type == 'None':
            return 1.0

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        iter_per_cycle = self.total_iter / n_cycle
        current_ratio = 1.0 * (self.current_iter % iter_per_cycle) / iter_per_cycle
        if current_ratio < ratio:
            return start + (current_ratio / ratio) * (stop - start)
        else:
            return stop
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.args.num_epoch, eta_min=1e-6)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        # Training data
        self.history = {
            "train_loss": [],
            "mse_loss": [],
            "kl_loss": [],
            "beta": [],
            "tfr": [],
            "val_psnr": []
        }
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            if i == 0:
                self.kl_annealing.total_iter = self.args.num_epoch * len(train_loader)
            epoch_mse = 0
            epoch_kl = 0
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                adapt_TeacherForcing = True if random.random() < self.tfr else False
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, mse, kl = self.training_one_step(img, label, adapt_TeacherForcing)
                epoch_mse += mse
                epoch_kl += kl
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))

            self.history["train_loss"].append((epoch_mse + epoch_kl)/len(train_loader))
            self.history["mse_loss"].append(epoch_mse/len(train_loader))
            self.history["kl_loss"].append(epoch_kl/len(train_loader))
            self.history["tfr"].append(self.tfr)
            self.history["beta"].append(self.kl_annealing.get_beta())
            if self.current_epoch % 1 == 0:
                self.save_history()

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        all_psnr = []
        all_loss = []
        gif_frames = []
        for idx, (img, label) in enumerate(pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, avg_psnr, frames  = self.val_one_step(img, label)
            all_loss.append(loss.item())
            all_psnr.append(avg_psnr)
            
            # 取第一支影片做視覺化
            if idx == 0:
                gif_frames = frames
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            pbar.set_postfix(loss=loss.item(), avg_psnr=np.mean(all_psnr))
        final_psnr = np.mean(all_psnr)
        print(f'\n--- Epoch {self.current_epoch} Validation Result ---')
        print(f'Average Loss: {np.mean(all_loss):.4f}')
        print(f'Average PSNR: {final_psnr:.4f}')
        print('-------------------------------------------\n')
        
        self.history["val_psnr"].append(final_psnr)

        # 儲存 GIF 供觀察
        if self.current_epoch % self.args.per_save == 0:
            gif_path = os.path.join(self.args.save_root, f'val_epoch_{self.current_epoch}.gif')
            self.make_gif(gif_frames, gif_path)
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)

        total_mse = 0
        total_kl = 0

        out_frame = img[0]

        for i in range(1, self.train_vi_len):
            
            image_features = self.frame_transformation.forward(img[i])
            label_features = self.label_transformation.forward(label[i])
            z, mu, logvar = self.Gaussian_Predictor.forward(image_features, label_features)
            logvar = torch.clamp(logvar, -10, 10) 
            mu = torch.clamp(mu, -10, 10)
            out_frame_feature = self.frame_transformation(out_frame)
            pred_frame = self.Generator(self.Decoder_Fusion(out_frame_feature, label_features, z))
            
            total_mse += self.mse_criterion(pred_frame, img[i])
            kl = kl_criterion(mu, logvar, img.shape[1])
            kl = torch.clamp(kl, max=1000000.0)
            total_kl += kl

            if adapt_TeacherForcing:
                out_frame = img[i]
            else:
                out_frame = torch.clamp(pred_frame, 0, 1).detach() 

        beta = self.kl_annealing.get_beta()# * 0.5
        self.kl_annealing.update()
        total_loss = total_mse + beta * total_kl

        self.optim.zero_grad()
        total_loss.backward()
        self.optimizer_step()

        return total_loss / (self.train_vi_len - 1), total_mse.item(), total_kl.item()
    
    def val_one_step(self, img, label):
        # TODO
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)

        total_mse = 0
        total_kl = 0
        pred_frames = []

        out_frame = img[0]
        pred_frames.append(out_frame[0].cpu())
        psnr_list = []

        for i in range(1, self.val_vi_len):
            
            #image_features = self.frame_transformation.forward(img[i])
            label_features = self.label_transformation.forward(label[i])
            z = torch.randn(1, self.args.N_dim).to(self.args.device)
            h_feat, w_feat = label_features.size(2), label_features.size(3)
            z = z.view(1, self.args.N_dim, 1, 1)
            z = z.expand(1, self.args.N_dim, h_feat, w_feat)
            out_frame_feature = self.frame_transformation(out_frame)
            pred_frame = self.Generator(self.Decoder_Fusion(out_frame_feature, label_features, z))
            pred_frame = torch.clamp(pred_frame, 0, 1) 
            psnr = Generate_PSNR(pred_frame, img[i])
            psnr_list.append(psnr.item())
            
            total_mse += self.mse_criterion(pred_frame, img[i])

            out_frame = pred_frame
            pred_frames.append(pred_frame[0].detach().cpu())

        total_loss = total_mse
        if self.current_epoch == self.args.num_epoch - 1:
            np.save(os.path.join(self.args.save_root, "psnr_per_frame.npy"), np.array(psnr_list))
        avg_psnr = np.mean(psnr_list)

        return total_loss / (self.val_vi_len - 1), avg_psnr, pred_frames
    
    def save_history(self):
        import json
        save_path = os.path.join(self.args.save_root, "history.json")
        with open(save_path, 'w') as f:
            json.dump(self.history, f)
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch >= self.args.tfr_sde:
            self.tfr -= self.args.tfr_d_step
            if self.tfr <= 0.0:
                self.tfr = 0.0
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optim.state_dict(),
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=8)
    parser.add_argument('--lr',            type=float,  default=0.0005,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=4,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.25,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
