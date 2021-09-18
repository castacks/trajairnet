import torch
from torch import nn

from model.tcn_model import TemporalConvNet
from model.gat_model import GAT
from model.cvae_base import CVAE
from model.utils import acc_to_abs




class TrajAirNet(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,n_heads,alpha,social=True,context=True):
        super(TrajAirNet, self).__init__()

        
        self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn_encoder_y = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.cvae = CVAE(encoder_layer_sizes = [144,128,128],latent_size = 128, decoder_layer_sizes = [128,96,96],conditional=True, num_labels= 278)
        self.gat = GAT( nin=139, nhid = 256, nout = 139,alpha = alpha,nheads = n_heads)
        self.linear_decoder = nn.Linear(32,12)
        self.context_conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2)
        self.context_linear = nn.Linear(10,7)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.linear_decoder.weight.data.normal_(0, 0.05)
        self.context_linear.weight.data.normal_(0, 0.05)
        self.context_conv.weight.data.normal_(0, 0.1)
        
    def forward(self, x, y, adj,context,sort=False):        
        
        encoded_trajectories_x = []
        encoded_appended_trajectories_x = []
        encoded_trajectories_y = []
        
        # pass all agents through encoder

        for agent in range(x.shape[2]):
            
            x1 = torch.transpose(x[:,:, agent][None, :, :], 1, 2)
            encoded_x = self.tcn_encoder_x(x1)
            encoded_x = torch.flatten(encoded_x)[None,None,:]
            encoded_trajectories_x.append(encoded_x)
            
            c1 = torch.transpose(context[:,:, agent][None, :, :], 1, 2)
            encoded_context = self.context_conv(c1)
            encoded_context = self.relu(self.context_linear(encoded_context))
            
            appended_x = torch.cat((encoded_x,encoded_context),dim=2)
            encoded_appended_trajectories_x.append(appended_x)
            
            y1 = torch.transpose(y[:,:, agent][None, :, :], 1, 2)
            encoded_y = self.tcn_encoder_y(y1)
            encoded_y = torch.flatten(encoded_y)[None,None,:]
            encoded_trajectories_y.append(encoded_y)

        gat_input = torch.squeeze(torch.stack(encoded_appended_trajectories_x,dim=2))
    
        
        if len(gat_input.shape) == 1:
            gat_input = torch.unsqueeze(gat_input,dim=0)

        gat_output = self.gat(gat_input,adj)

        recon_y = []
        m = []
        var = []
        
        # pass all agents through decoder
        for agent in range(x.shape[2]):
            
            H_x = gat_output[agent].unsqueeze(0).unsqueeze(0)
            H_xx = encoded_appended_trajectories_x[agent]
            H_x = torch.cat((H_xx,H_x),dim=2)


            H_y = encoded_trajectories_y[agent]
            H_yy, means,log_var, z = self.cvae(H_y,H_x)

            H_yy =  torch.reshape(H_yy, (3, -1))
            recon_y_x = (self.linear_decoder(H_yy))
            recon_y_x = torch.unsqueeze(recon_y_x,dim=0)
            recon_y_x = acc_to_abs(recon_y_x,x[:,:,agent][:,:,None])    

            recon_y.append(recon_y_x)
            m.append(means)
            var.append(log_var)
        return recon_y,m,var
    
    
    def inference(self,x,z,adj,context):
     

        encoded_trajectories_x = []
        encoded_appended_trajectories_x = []
        
        # pass all agents through encoder
        for agent in range(x.shape[2]):
            x1 = torch.transpose(x[:,:, agent][None, :, :], 1, 2)
            c1 = torch.transpose(context[:,:, agent][None, :, :], 1, 2)
            encoded_context = self.context_conv(c1)
            encoded_context = self.relu(self.context_linear(encoded_context))

            encoded_x = self.tcn_encoder_x(x1)
            encoded_x = torch.flatten(encoded_x)[None,None,:]
            encoded_trajectories_x.append(encoded_x)
            appended_x = torch.cat((encoded_x,encoded_context),dim=2)

            encoded_appended_trajectories_x.append(appended_x)

        gat_input = torch.squeeze(torch.stack(encoded_appended_trajectories_x,dim=2))
        
        if len(gat_input.shape) == 1:
            gat_input = torch.unsqueeze(gat_input,dim=0)
        
        gat_output = self.gat(gat_input,adj)
        
        recon_y = []
        m = []
        var = []
        
        # pass all agents through decoder
        for agent in range(x.shape[2]):
            H_x = (gat_output[agent].unsqueeze(0)).unsqueeze(0)
            H_xx = encoded_appended_trajectories_x[agent]
            H_x = torch.cat((H_xx,H_x),dim=2)
            H_yy = self.cvae.inference(z,H_x)
            H_yy =  torch.reshape(H_yy, (3, -1))

            recon_y_x = (self.linear_decoder(H_yy)) 
            recon_y_x = torch.unsqueeze(recon_y_x,dim=0)
            recon_y_x = acc_to_abs(recon_y_x,x[:,:,agent][:,:,None])    

            recon_y.append(recon_y_x.squeeze().detach())
     
        return recon_y