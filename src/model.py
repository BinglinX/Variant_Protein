import torch.nn.functional as F
import torch.nn as nn
import torch

class CustomMLP(nn.Module):
    '''
    input: a dictionary in the following form:
                {"wt":{"embed":wt_emb_padded,"mask":wt_mask},
                 "mt":{"embed":mt_emb_padded,"mask":mt_mask},
                }
            a list of label

    each of the embeddings first goes through 2 layers:
    hidden layer(dropout+activation+linear): (batch size, max_length, num_feature) -> (batch size, max_length, num_hidden)
    mask_normalise layer: (batch size, max_length, num_hidden) -> (batch_size,num_hidden)

    wt and mt are then concatenated together: 
    (batch_size, num_hidden) -> (batch_size, num_hidden*2)

    predict layer(dropout+activation+linear): (batch_size, num_hidden*2) -> (batch_size, num_output)
    fc layer: (batch_size, output) -> (batch_size,1)
    '''

    def __init__(self, cfg):
        super(CustomMLP, self).__init__()
        '''
        defines parameters and layers of the model
        '''

        #dimension of feature per residue, input shape of the first linear layer
        num_input = cfg.num_feature

        #output shape of first and second linear layer
        num_hidden, num_output = cfg.num_hidden, cfg.num_output 

        #dropout of first and second linear layer
        dropout_hidden, dropout_predict= cfg.dropout_hidden, cfg.dropout_predict 
        
        #defines a dictionary of activation functions
        ac_fn_dict = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU
        } 

        #activation function of first and second linear layer
        ac_fn_hidden, ac_fn_predict = ac_fn_dict[cfg.ac_fn_hidden], ac_fn_dict[cfg.ac_fn_predict]

        #first linear layer: (batch size, max_length, num_feature) -> (batch size, max_length, num_hidden)
        self.hidden = nn.Sequential(
            nn.Dropout(dropout_hidden),
            ac_fn_hidden(inplace=True),
            nn.Linear(num_input,num_hidden),
        )

        #second linear layer: (batch_size, num_hidden*2) -> (batch size, num_output)
        self.predict = nn.Sequential(
            nn.Dropout(dropout_predict),
            ac_fn_predict(inplace=True),
            nn.Linear(num_hidden*2,num_output),
        )

        #final fully connected layer
        self.fc = nn.Linear(num_output,1)

    def mask_normalise(self, x, mask):

        '''
        removes the padding by multiplying with mask, then normalise along the sequence by taking average
        '''

        mask_expanded = mask.unsqueeze(-1).expand_as(x) #(batch_size,max_length)->(batch_size,max_length,1)->(batch_size,max_length,num_hidden)
        x = x * mask_expanded
        sum_mask = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1) #(batch_size,max_length,num_hidden), sums along the max_length dimension
                                                                       #clamp to avoid division by 0
        cls = x.sum(dim=1, keepdim=True) / sum_mask
        return cls.squeeze(1) #(batch_size,max_length,num_hidden) -> #(batch_size,num_hidden)

    def forward(self,x):
        '''
        first retrieves embedding and data
        x is a dictionary with the following form:

                {"wt":{"embed":wt_emb_padded,"mask":wt_mask},
                 "mt":{"embed":mt_emb_padded,"mask":mt_mask}
                }
        gets wt from x["wt"], mt from x["mt"]
        '''

        wt, mt = x["wt"]["embed"], x["mt"]["embed"]

        wt_mask, mt_mask = x["wt"]["mask"], x["mt"]["mask"]

        #first linear layer: hidden layer (batch size, max_length, num_feature) -> (batch size, max_length, num_hidden)
        wt, mt = self.hidden(wt), self.hidden(mt)

        #mask_normalise: (batch size, max_length, num_hidden) -> (batch size, num_hidden)
        wt, mt = self.mask_normalise(wt,wt_mask), self.mask_normalise(mt,mt_mask)

        #concatenation: (batch_size, num_hidden) -> (batch_size, num_hidden*2)
        x = torch.cat([wt,mt],dim=1) #concat along the num_hidden dimension

        #second linear layer: (batch_size, num_hidden*2) -> (batch size, num_output)
        x = self.predict(x)

        #final fully connected layer: (batch size, num_output) -> (batch size, 1)
        x = self.fc(x)

        return x



   