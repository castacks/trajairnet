import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from model.trajairnet import TrajAirNet
from model.utils import ade, fde, TrajectoryDataset

def main():
    
    parser=argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--obs',type=int,default=11)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)
    
    args=parser.parse_args()


    ##Select device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##Load data

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)
    loader_test = DataLoader(dataset_test,batch_size=1,num_workers=4,shuffle=True,collate_fn=seq_collate)

    ##Load model

    input_channels = args.input_channels
    n_classes = int(args.preds/args.preds_step)
    channel_sizes= [args.tcn_channel_size]*args.tcn_layers
    channel_sizes.append(n_classes)
    kernel_size = args.tcn_kernels
    dropout = args.dropout
    lr = args.lr
    graph_hidden = args.graph_hidden
    n_heads = args.gat_heads 
    alpha = args.alpha

    model = TrajAirNet(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout,n_heads=n_heads,alpha=alpha)
    model.to(device)
    

def test(model,loader_test,device):
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_batch = 0
    for batch in tqdm(loader_test):
        tot_batch += 1
        if tot_batch>8:
            break 
        batch = [tensor.to(device) for tensor in batch]

        obs_traj_all , pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, seq_start  = batch
        num_agents = obs_traj_all.shape[1]
        
        best_ade_loss = float('inf')
        best_fde_loss = float('inf')
        
        for i in range(5):
            z = torch.randn([1,1 ,128]).to(device)
            
            adj = torch.ones((num_agents,num_agents))
            recon_y_all = model.inference(torch.transpose(obs_traj_all,1,2),z,adj,torch.transpose(context,1,2))
            
            ade_loss = 0
            fde_loss = 0
            for agent in range(num_agents):
                obs_traj = np.squeeze(obs_traj_all[:,agent,:].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[:,agent,:].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()
                ade_loss += ade(recon_pred, pred_traj)
                fde_loss += fde((recon_pred), (pred_traj))
           
            
            ade_total_loss = ade_loss/num_agents
            fde_total_loss = fde_loss/num_agents
            if ade_total_loss<best_ade_loss:
                best_ade_loss = ade_total_loss
                best_fde_loss = fde_total_loss

        tot_ade_loss += best_ade_loss
        tot_fde_loss += best_fde_loss
    return tot_ade_loss/(tot_batch),tot_fde_loss/(tot_batch)


if __name__=='__main__':
    main()

