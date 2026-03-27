import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from typing import Dict, List
from copy import deepcopy

import mrlfads.paths as path
from mrlfads.utils.torch_utils import Conv1dBase

class VariableDecoder(pl.LightningModule):
    """
    Decodes variables (e.g. behavior, task variables) from inferred quantities
    of trained MRLFADS module.
    """
    def __init__(
        self,
        model,
        predictions: list,
        variable: str,
        criterion,
        
        session: int = 0,
        lr: float = 4.0e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore = ['model', 'criterion'])
        
        self.model = model
        self._build_areas()
        
        # e.g. nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.criterion = criterion
        
    def forward(self, batch):
        with torch.inference_mode():
            self.model.eval()
            self.model.predict_step(batch, 0)
        
        # Pass inference through decoders
        outputs = {}
        for str_tuple, decoder in self.decoders.items():
            key_tuple = self.rmap_key(str_tuple)
            inf_type = key_tuple[0]
            slice_func = getattr(self, f'_get_{inf_type}')
            quant = slice_func(key_tuple[1:])
            quant = quant.permute(0, 2, 1)
            outputs[str_tuple] = decoder(quant).permute(0, 2, 1)
        return outputs
    
    def _shared_step(self, batch, batch_idx, step_type):
        hps = self.hparams
        self.outputs = self(batch)
        
        # Forward
        loss = 0.0
        metrics = {}
        start = self.model.hparams.ic_enc_seq_len
        self.targets = self.model.current_info[hps.session][hps.variable][:, start:]
        for key_tuple in hps.predictions:
            str_tuple = self.map_key(key_tuple)
            lval = self.criterion(
                    input=self.outputs[str_tuple],
                    target=self.targets,
            )
            loss += lval
            metrics[f"{step_type}/loss_{str_tuple}"] = lval
            
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                f1 = self.f1_score(self.outputs[str_tuple], self.targets)
                metrics[f"{step_type}/acc_{str_tuple}"] = f1
            
        metrics[f"{step_type}/loss"] = loss
        
        if step_type != 'predict':
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
            )
            
        return loss
            
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")
        
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "valid")
        
    def predict_step(self, batch, batch_idx, sample=False):
        return self._shared_step(batch, batch_idx, "predict")
    
    def configure_optimizers(self):
        hps = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = hps.lr,
        )
        return optimizer
            
    def _get_rates(self, area_name):
        hps = self.hparams
        return self.model.outputs[area_name[0]][hps.session]
    
    def _get_messages(self, pair):
        src, tar = pair
        src = src.strip("(')")
        tar = tar.strip("(')").strip(" '")
        ahps = self.model.areas[tar].hparams
        ms = self.model.save_var[tar].inputs[..., ahps.ci_enc_dim:-ahps.co_dim]
        ms_split = torch.split(ms, [ahps.com_dim] * (len(self.model.areas)-1), dim=2)
        other_area_names = deepcopy(self.model.area_names)
        other_area_names.remove(tar)
        idx = other_area_names.index(src)
        return self.split_by_batch(tar, ms_split[idx])
    
    def _get_inputs(self, area_name):
        ahps = self.model.areas[area_name[0]].hparams
        inputs = self.model.save_var[area_name[0]].inputs[..., -ahps.co_dim:]
        return self.split_by_batch(area_name[0], inputs)
    
    def _get_factors(self, area_name):
        ahps = self.model.areas[area_name[0]].hparams
        factors = self.model.save_var[area_name[0]].states[:, 1:, -ahps.fac_dim:]
        return self.split_by_batch(area_name[0], factors)

    def _build_areas(self,):
        hps = self.hparams
        
        self.decoders = nn.ModuleDict()
        for pred_tuple in hps.predictions:
            if pred_tuple[0] == 'rates':
                area_name = pred_tuple[1]
                input_size = self.model.areas[area_name].hparams.num_neurons[hps.session]
                
            elif pred_tuple[0] == 'messages':
                src_area, tar_area = pred_tuple[1]
                input_size = self.model.areas[tar_area].hparams.com_dim
            
            elif pred_tuple[0] == 'inputs':
                area_name = pred_tuple[1]
                input_size = self.model.areas[area_name].hparams.co_dim
                
            elif pred_tuple[0] == 'factors': 
                area_name = pred_tuple[1]
                input_size = self.model.areas[area_name].hparams.fac_dim
                
            else:
                raise ValueError()
                
            output_size = 1 # Decode one variable per model
            self.decoders[self.map_key(pred_tuple)] = Conv1dBase(
                [[input_size, 16, 11, 'ReLU'],
                 [16, 8, 11, 'ReLU'],
                 [8, output_size, 1, ''],
                ]
            )
            
    def split_by_batch(self, area_name, arr):
        batch_sizes = []
        for key in self.model.raw_batch.keys():
            batch_sizes.append( self.model.raw_batch[key].encod_data[area_name].shape[0] )
        return torch.split(arr, batch_sizes, dim=0)[self.hparams.session]
    
    @staticmethod
    def map_key(tup): return ','.join([str(i) for i in tup])

    @staticmethod
    def rmap_key(string): return string.split(',')

    @staticmethod
    @torch.no_grad()
    def f1_score(
        logits: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-8
    ) -> torch.Tensor:
        pred = (torch.sigmoid(logits) >= threshold).to(target.dtype)
        tgt = (target > 0.5).to(target.dtype)
        pred = pred.reshape(-1)
        tgt = tgt.reshape(-1)

        tp = (pred * tgt).sum()
        fp = (pred * (1 - tgt)).sum()
        fn = ((1 - pred) * tgt).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return f1
                                               
