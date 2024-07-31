import os
import torch
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.model import *
from src.model_module import TrainingModule
from src.dataset_module import ProteinVariant_DataModule

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import wandb
from scipy.stats import spearmanr,pearsonr,linregress


class ModelTrainer(object):
    def __init__(self, cfg: HydraConfig):
        self.cfg = cfg

        """ model parameters """
        os.makedirs(os.path.dirname(cfg.model.model_save_path), exist_ok=True)
        self.model_checkpoint = os.path.join(self.cfg.model.model_save_path, f"{self.cfg.model.model_save_filename}.ckpt")

        self.define_logging()
        self.load_device()

        if cfg.general.usage in ("train", "infer"):
            self.load_data()
            self.load_model()
            self.define_trainer()
    
    def define_logging(self):
        log_file_dir = os.path.dirname(self.cfg.general.save_path_log)
        os.makedirs(log_file_dir, exist_ok=True)

        # 配置根日志器
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %A %H:%M:%S',
            filename=self.cfg.general.save_path_log,
            filemode='w'
        )

        # 配置控制台日志处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    def load_wandb(self):
        # Wandb
        cfg = self.cfg
        wandb_run_id = cfg.wandb.run_id if cfg.wandb.run_id else wandb.util.generate_id()
        self.wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.run_name}_{cfg.general.save_num}",
            id=wandb_run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    def load_device(self):
        if torch.cuda.is_available():
            logging.info(
                f'There are {torch.cuda.device_count()} GPU(s) available.')
            logging.info(f'Device name: {torch.cuda.get_device_name(0)}')
        else:
            logging.info("////////////////////////////////////////////////")
            logging.info("///// NO GPU DETECTED! Falling back to CPU /////")
            logging.info("////////////////////////////////////////////////")
    
    def load_model(self):
        config = self.cfg.model

        logging.info("Loading model...")

        model_mgr = {"mlp": CustomMLP}

        self.model_object = model_mgr[config.model_choice](config)
        self.model = TrainingModule(self.model_object, cfg=config)

        logging.info(f"Model {config.model_choice} loaded.\n")
        logging.info(f"Optimizer {config.optimizer} used.\n")
        logging.info(f"Loss_function {config.loss_fn} used.\n")
    
    def load_data(self):
        logging.info(
            f"Loading data for the purpose of ** {self.cfg.general.usage} ** ...")
        logging.info(
            f'Loading data from {self.cfg.dataset.load_data.ds_path}...')
        
        self.data_module = ProteinVariant_DataModule(self.cfg.dataset.load_data)

    def define_trainer(self):
        cfg = self.cfg
         # ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            monitor='valid_loss', # valid_loss, none
            dirpath=cfg.model.model_save_path,
            filename=cfg.model.model_save_filename,
            save_top_k=1,
            mode='min',
            save_last=False,
        )

        # EarlyStopping
        early_stop_callback = EarlyStopping(
            monitor='valid_loss',
            patience=cfg.model.early_stop,
            verbose=True,
            mode='min'
        )

        if not self.cfg.model.debug:
            self.load_wandb()
        else:
            self.wandb_logger = None

        # Trainer
        self.trainer = pl.Trainer(
            logger=self.wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            max_epochs=cfg.model.n_epochs,
            accelerator='gpu',
            devices=cfg.general.devices, 
            accumulate_grad_batches=cfg.model.grad_accum_steps,
            precision=32,
            check_val_every_n_epoch=1
        )

    def train(self):
       
        if os.path.exists(self.model_checkpoint):
            model_checkpoint = self.model_checkpoint
            logging.info(f"Checkpoint Found, Loading from: {self.model_checkpoint}...")
        else:
            model_checkpoint = None


        self.trainer.fit(self.model, datamodule=self.data_module, ckpt_path=model_checkpoint)

        best_model = TrainingModule.load_from_checkpoint(checkpoint_path=self.model_checkpoint, model=self.model_object, cfg=self.cfg.model)
        self.trainer.test(model=best_model, datamodule=self.data_module)
        
        output = best_model.test_result["test_preds"]
        label = best_model.test_result["test_labels"]      

        run_metrics(output,label)

    def predict(self):

        trained_model = TrainingModule.load_from_checkpoint(checkpoint_path=self.model_checkpoint, model=self.model_object, cfg=self.cfg.model)
        predictions = self.trainer.predict(model=trained_model, datamodule=self.data_module)


        #prepares prediction and its index
        keys = self.data_module.target_key_list #the target_key_list has been made an attribute for accessing
        output = torch.cat([item['pred'] for item in predictions], dim= 0)

        #zip the index and the output together
        pred = list(zip(keys,output))

        torch.save(pred, f"{self.cfg.general.save_path_predictions}/{self.cfg.general.save_name}.pt")
        logging.info(f"Prediction results saved at {self.cfg.general.save_path_predictions}")

def run_metrics(output,label):
    '''
    calculates metrics including spearman, pearson and r_squared and returns as a dictionary
    used in validation and test
    '''
    
    #convert the prediction and labels to numpy arrays and flatten them for statistical tests
    output_np = output.numpy().flatten()
    label_np = label.numpy().flatten()

    #pearsonr and spearmanr produces a Result object, with attributes "statistic" and "pvalue"
    pearson = pearsonr(output_np,label_np).statistic 
    spearman = spearmanr(output_np,label_np).statistic 

    #lingress gives an r value, which should be squared to give the r squared value
    r_squared = (linregress(output_np,label_np).rvalue)^2


    logging.info(f"Spearman correlation: {spearman} Pearson correlation: {pearson},r_squared: {r_squared}")

@hydra.main(version_base=None, config_path="./config", config_name="base")
def main(cfg: HydraConfig) -> None:
    pl.seed_everything(cfg.general.seed)

    mutable_cfg = OmegaConf.to_container(cfg, resolve=True)
    mutable_cfg = OmegaConf.create(mutable_cfg)

    model_trainer = ModelTrainer(mutable_cfg)
    if cfg.general.usage == 'train':
        model_trainer.define_trainer()
        model_trainer.train()

    elif cfg.general.usage == 'infer':
        model_trainer.predict()


if __name__ == "__main__":
    main()
