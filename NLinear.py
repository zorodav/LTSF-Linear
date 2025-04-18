
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self, train_data = None).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.train_data = train_data
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
        

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]

    def predict(self, test_data):
        """
        Generate predictions for the given test data.
    
        Args:
            test_data (torch.Tensor): Input data of shape [Batch, Input length, Channel].
    
        Returns:
            torch.Tensor: Predictions of shape [Batch, Output length, Channel].
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for inference
            predictions = self.forward(test_data)  # Pass test data through the forward method
        return predictions        
