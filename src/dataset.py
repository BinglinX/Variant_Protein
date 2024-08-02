import torch
from torch.utils.data import Dataset
from src.lmdb_loader import LMDBLoader

class ProteinVariant_Dataset(Dataset):
    def __init__(self, key_list, label_list, wt_emb_dict, cfg):
        self.key_list = key_list #list of mt keys
        self.label = label_list #list of labels(DMS score)
        self.wt_emb_dict= wt_emb_dict #pt saving the wt embedding
        self.cfg = cfg
        lmdb_path = cfg.lmdb_path #lmdb path saving the mt embedding
        self.lmdb_loader=LMDBLoader(lmdb_path,self.key_list)

    def __len__ (self):
        return len(self.key_list)
    
    def __getitem__ (self, index):
        '''
        from an index, retrieves the corresponding embeddings of wt and mt, pads them and returns the
        data in a dictionary and the labels in a list
        '''

        wt_id = self.key_list[index].split("-")[0] #mt are in the form of "wt-mtid", splitting with "-" to get wt
        wt_emb = self.wt_emb_dict[wt_id]
        mt_emb = self.lmdb_loader[index]

        wt_emb_padded, wt_mask = self.pad_tensor(wt_emb)
        mt_emb_padded, mt_mask = self.pad_tensor(mt_emb)

        return({"wt":{"embed":wt_emb_padded,"mask":wt_mask},
                 "mt":{"embed":mt_emb_padded,"mask":mt_mask}},
                self.label[index]
              )

    def pad_tensor(self, tensor):
        '''
        pads all tensors to the max_tensor length
        input tensor: 2D, shape = (seq_len, feature_dim)
        padding: 2D, shape = (max_tensor_length - seq_len, feature_dim)
        '''

        max_tensor_length = self.cfg.max_tensor_length

        seq_len, feature_dim = tensor.shape
        if seq_len < max_tensor_length:

            padding_size = max_tensor_length - seq_len

            padding = torch.zeros(
                (padding_size, feature_dim), dtype=tensor.dtype)
            padded_tensor = torch.cat([tensor, padding], dim=0)

            mask = torch.cat([torch.ones(seq_len, dtype=torch.long),
                            torch.zeros(padding_size, dtype=torch.long)], dim=0)
        else:
 
            padded_tensor = tensor[:max_tensor_length]

            mask = torch.ones(max_tensor_length, dtype=torch.long)

        return padded_tensor, mask



    

        
        








