import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from model.trajairnet import TrajAirNet
from model.utils import ade, fde, TrajectoryDataset, seq_collate

def main():
    
    parser=argparse.ArgumentParser(description='Test TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')

    parser.add_argument('--obs',type=int,default=11)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)

    parser.add_argument('--input_channels',type=int,default=3)
    parser.add_argument('--tcn_channel_size',type=int,default=256)
    parser.add_argument('--tcn_layers',type=int,default=2)
    parser.add_argument('--tcn_kernels',type=int,default=4)


    parser.add_argument('--gat_heads',type=int, default=16)
    parser.add_argument('--graph_hidden',type=int,default=8)
    parser.add_argument('--dropout',type=float,default=0.05)
    parser.add_argument('--alpha',type=float,default=0.2)



    parser.add_argument('--delim',type=str,default=' ')

    parser.add_argument('--model_dir', type=str , default="/saved_models/")
    
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

    model_path =  os.getcwd() + args.model_dir + "model_" + args.dataset_name + "_" + str(args.epoch) + ".pt"


    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_ade_loss, test_fde_loss = test(model,loader_test,device)

    print("Test ADE Loss: ",test_ade_loss,"Test FDE Loss: ",test_fde_loss)



def test(model,loader_test,device):
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_batch = 0
    for batch in tqdm(loader_test):
        tot_batch += 1
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

