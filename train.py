import argparse
import os 
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from torch import optim



from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate, loss_func
from test import test



def train():

    ##Dataset params
    parser=argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--obs',type=int,default=11)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)

    ##Network params
    parser.add_argument('--input_channels',type=int,default=3)
    parser.add_argument('--tcn_channel_size',type=int,default=256)
    parser.add_argument('--tcn_layers',type=int,default=2)
    parser.add_argument('--tcn_kernels',type=int,default=4)


    parser.add_argument('--gat_heads',type=int, default=16)
    parser.add_argument('--graph_hidden',type=int,default=8)
    parser.add_argument('--dropout',type=float,default=0.05)
    parser.add_argument('--alpha',type=float,default=0.2)

    parser.add_argument('--lr',type=float,default=0.0001)


    parser.add_argument('--max_epoch',type=int, default=50)
    parser.add_argument('--delim',type=str,default=' ')
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)

    parser.add_argument('--model_pth', type=str , default="/saved_models/")

    args=parser.parse_args()


    ##Select device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##Load test and train data
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Train Data from ",datapath + "train")
    dataset_train = TrajectoryDataset(datapath + "train", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    loader_train = DataLoader(dataset_train,batch_size=1,num_workers=4,shuffle=True,collate_fn=seq_collate)
    loader_test = DataLoader(dataset_test,batch_size=1,num_workers=4,shuffle=True,collate_fn=seq_collate)

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

    ##Resume
    # checkpoint = torch.load('model_11.pt',map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(model.parameters(),lr=lr)

    num_batches = len(loader_train)
 
    print("Starting Training....")

    for epoch in range(1, args.max_epoch+1):

        model.train()
        loss_batch = 0 
        batch_count = 0
        tot_batch_count = 0
        tot_loss = 0
        for batch in tqdm(loader_train):
            if tot_batch_count > 16:
                break
            batch_count += 1
            tot_batch_count += 1
            batch = [tensor.to(device) for tensor in batch]
            obs_traj , pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch 
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj,1,2)
            adj = torch.ones((num_agents,num_agents))

            optimizer.zero_grad()
            recon_y,m,var = model(torch.transpose(obs_traj,1,2),pred_traj, adj[0],torch.transpose(context,1,2))
            loss = 0
            
            for agent in range(num_agents):
                loss += loss_func(recon_y[agent],torch.transpose(pred_traj[:,:,agent],0,1).unsqueeze(0),m[agent],var[agent])
            
            loss_batch += loss
            tot_loss += loss.item()
            if batch_count>8:
                loss_batch.backward()
                optimizer.step()
                loss_batch = 0 
                batch_count = 0

        print("EPOCH: ",epoch,"Train Loss: ",loss)

        if args.save_model:  
            loss = tot_loss/tot_batch_count
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            print("Saving model at",model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, model_path)
        
        if args.evaluate:
            print("Starting Testing....")
        
            model.eval()
            test_ade_loss, test_fde_loss = test(model,loader_test,device)

            print("EPOCH: ",epoch,"Train Loss: ",loss,"Test ADE Loss: ",test_ade_loss,"Test FDE Loss: ",test_fde_loss)

if __name__=='__main__':

    train()