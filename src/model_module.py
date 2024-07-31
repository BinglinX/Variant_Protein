import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.stats import spearmanr,pearsonr,linregress


class TrainingModule(pl.LightningModule):
    def __init__(self,model,cfg):
        super(TrainingModule, self).__init__()
        '''
        cfg comes from the model part of config
        '''

        self.model = model
        self.cfg = cfg

        #define a metrics dictionary and a loss list for calculation of average in on_valid_epoch_end
        self.val_step_metrics = {"spearman":[],"pearson":[],"r_squared":[]}
        self.val_step_losses = []


        self.test_step_output = {'pred': [], "label": []}
        self.predict_step_output = {'pred': [], "label": []}

    def training_step(self,batch,batch_idx):
        '''
        The training step calculates the loss and the metrics, saves them
        and returns the loss for training
        Each batch is a tuple of a dictionary and a list

        dictionary: 
        {"wt":{"embed":wt_emb_padded,"mask":wt_mask},
               "mt":{"embed":mt_emb_padded,"mask":mt_mask}},
                
        list: label(DMS scores)
            
        '''

        input,label = batch
        output = self.model(input) #model takes the dictionary as input and returns a single value

        loss = self.calculate_loss(output,label) #calculates MSEloss

        metrics = self.calculate_metrics(output,label) #calculates metrics including spearman correlation, pearson correlation and r squared
        spearman, pearson, r_squared = metrics["spearman"],metrics["pearson"],metrics["r_squared"]

        #defines a record dict and use the _logger method to log them
        record_dict = {"loss":loss,"spearman":spearman,"pearson":pearson,"r_squared":r_squared}
        self._logger(record_dict,"train")

        return loss
    
    def validation_step(self,batch,batch_idx):
        '''
        The validation step calculates the loss and the scores, saves the loss and scores,
        and returns the loss for early stop monitoring
        '''

        input,label = batch
        output = self.model(input) #model takes the dictionary as input and returns a single value

        loss = self.calculate_loss(output,label) #calculates MSEloss and stores it
        self.val_step_losses.append(loss)

        metrics = self.calculate_metrics(output,label) #calculates metrics including spearman correlation, pearson correlation and r squared

        #store the metrics in the dictionary
        for key in self.val_step_metrics.keys():
            self.val_step_metrics[key].append(metrics[key])

        #defines a record dict and use the _logger method to log them
        record_dict = {"loss":loss,"spearman":metrics["spearman"],
                       "pearson":metrics["pearson"],"r_squared":metrics["r_squared"]}

        self._logger(record_dict,"valid")


    def test_step(self,batch,batch_idx):
        '''
        The test step gets the prediction
        and saves the predictions and true labels
        '''

        input,label = batch
        output = self.model(input)

        self.test_step_output['pred'].append(output)
        self.test_step_output['label'].append(label)

    def predict_step(self,batch,batch_idx):
        '''
        The predict step calculates the model's predictions and stores them for further analysis
        '''
        input, label = batch #label should be a placeholder here and does not have a meaning, might be removed in further updates
        output = self.model(input)

        self.predict_step_output['pred'].append(output)
        self.predict_step_output['label'].append(label)   

        return {"pred": output}

    def on_validation_epoch_end(self):
        '''
        calculates an average of the metrics and log them
        '''

        #defines a dictionary of metrics
        val_epoch_avg_metrics =  {"spearman":None,"pearson":None,"r_squared":None}

        #for each metrics, calculates its average over the epoch and stores it
        for key in self.val_step_metrics.keys():
            metrics = self.val_step_metrics[key] #each entry of the dictionary is a list of metrics
            val_epoch_avg_metrics[key] = sum(metrics)/len(metrics)
        
        #calculates the average of losses over the epoch and stores it
        val_epoch_avg_loss = sum(self.val_step_losses)/len(self.val_step_losses)

        #the epoch dict first copies the metric dictionary and then updates its self with the loss,
        #creating a dictionary with both the metrics and the loss
        epoch_dict = val_epoch_avg_metrics.copy()
        epoch_dict["loss"] = val_epoch_avg_loss
        
        self.log_dict(epoch_dict,on_step = False,
            on_epoch= True,
            prog_bar = True,
            batch_size = self.cfg.batch_size)
        
        self.val_step_metrics.clear()
        self.val_step_losses.clear()

    def on_test_epoch_end(self):
        '''
        prepares the prediction and the labels for calculation of metrics(see main.py)
        '''
        test_preds = torch.cat(self.test_step_output['pred'], dim=0)
        test_labels = torch.cat(self.test_step_output['label'], dim=0)

        self.test_step_output.clear()

        self.test_result =  {'test_preds': test_preds, 'test_labels': test_labels}

    
    def configure_optimizers(self):
        
        optimizer_dict = {
                "adam": torch.optim.Adam,
                "adamw": torch.optim.AdamW
        }
        return optimizer_dict[self.cfg.optimizer](self.model.parameters
        (), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def calculate_loss(self,output,label):
        '''
        uses the loss function written in the config to calculate loss and returns the loss
        used in training and validation
        '''

        loss_fn_dict = {"MSEloss":nn.MSEloss(reduction='mean')}
        loss_fn = loss_fn_dict[self.cfg.loss_fn]
        return loss_fn(output,label)
    
    def calculate_metrics(self,output,label):
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


        return {"spearman":spearman,"pearson":pearson,"r_squared":r_squared}
        

    def _logger(self, record_dict, dclass):
        for k,v in record_dict.items():
            name = f"{dclass}_{k}"
            self.log(name, v, 
            on_step = False,
            on_epoch= True,
            prog_bar = True,
            batch_size = self.cfg.batch_size
            )

    


