from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from src.dataset import ProteinVariant_Dataset
import pandas as pd



class ProteinVariant_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        '''
        cfg comes from the dataset.load_data part of config
        '''
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        '''
        Loads the embeddings and lists of keys
        Creates the following variables:
            wt_emb_dict: embeddings for wt
            key_list: keys of mt
            label_list: list of labels
        '''

        wt_emb_dict = torch.load(self.cfg.wt_emb)

        if stage == "fit" or stage is None:
            train_csv = pd.read_csv(self.cfg.train_sub_list)
            valid_csv = pd.read_csv(self.cfg.valid_sub_list)

            train_key_list = train_csv.loc[:,"mut_name"]
            valid_key_list = valid_csv.loc[:,"mut_name"]
            train_label_list = torch.tensor(train_csv.loc[:,"DMS_score"].values, dtype=torch.float32)
            valid_label_list = torch.tensor(valid_csv.loc[:,"DMS_score"].values, dtype=torch.float32)            

            self.train_dataset = ProteinVariant_Dataset(key_list=train_key_list,label_list=train_label_list,
                                                        wt_emb_dict=wt_emb_dict,cfg=self.cfg)
            self.valid_dataset = ProteinVariant_Dataset(key_list=valid_key_list,label_list=valid_label_list,
                                                        wt_emb_dict=wt_emb_dict,cfg=self.cfg)

        if stage == "test" or stage is None:
            test_csv = pd.read_csv(self.cfg.test_sub_list)
            test_key_list = test_csv.loc[:,"mut_name"]
            test_label_list = torch.tensor(test_csv.loc[:,"DMS_score"].values, dtype=torch.float32)


            self.test_dataset = ProteinVariant_Dataset(key_list=test_key_list,label_list=test_label_list,
                                                        wt_emb_dict=wt_emb_dict,cfg=self.cfg)

        if stage == "predict" or stage is None:
            target_csv = pd.read_csv(self.cfg.target_sub_list)
            target_key_list = target_csv.loc[:,"mut_name"] 
            target_label_list = torch.zeros((len(target_key_list),), dtype=torch.float32) #placeholder with no meaning

            self.predict_dataset = ProteinVariant_Dataset(key_list=target_key_list,label_list=target_label_list,
                                                        wt_emb_dict=wt_emb_dict,cfg=self.cfg)
            


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=10,
            pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10,
            pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10,
            persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10,
            persistent_workers=True)
