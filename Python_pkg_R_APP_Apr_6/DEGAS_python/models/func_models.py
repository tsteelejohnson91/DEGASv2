# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# import torch modules
import torch
import torch.nn as nn
# import loss functions
from .base_model import *
from .losses import *
import pandas as pd


##################################################################################
####################### Define Models ############################################
##################################################################################
                
class BlankCoxModel(BaseModel):
    def __init__(self, opt):
        super(BlankCoxModel, self).__init__(opt)
        
        # specify the feature extractor structures
        self.feature_extractor_layer = FeatureExtractor(opt).to(self.device)
        # specify neural network structures
        self.discrim_layer = Discriminator(opt).to(self.device)
        # add prediction layers
        self.low_reso_pred_layer = define_linear(opt["feature_dim"], opt["low_reso_output_shape"], opt["dropout_rate"], nn.Sigmoid()).to(self.device)
        # add all layer names
        self.net_name_list = ["feature_extractor_layer", "discrim_layer", "low_reso_pred_layer"]

        # add loss functions
        if self.loss_type == "LogNeg":
            self.criterion_low_reso = NegativeLogLikelihood(opt).to(self.device)
        elif self.loss_type == "CoxPH":
            self.criterion_low_reso = CoxPH(opt).to(self.device)
        else:
            raise ValueError("Loss function must be CoxPH or LogNeg!")

        # define loss types
        self.loss_names = ["low_reso_loss", "transfer_loss"]
        self.loss_rec = pd.DataFrame(dict(zip(self.loss_names, [[]] * len(self.loss_names))))
        
        # add optimizers
        self.optimizer1 = torch.optim.Adam(list(self.feature_extractor_layer.parameters()) + 
                                           list(self.low_reso_pred_layer.parameters()), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        self.optimizer2 = torch.optim.Adam(self.discrim_layer.parameters(), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        
        
        
    def set_input(self, input1, input2):
        self.high_reso_input = Variable(input1["data"].float().to(self.device))
        self.low_reso_input = Variable(input2["data"].float().to(self.device))
        self.low_reso_time = Variable(input2["time"].long().to(self.device))
        self.low_reso_status = Variable(input2["status"].float().to(self.device))
        

    def forward(self):
        _, self.emb3 = self.feature_extractor_layer(torch.concat((self.high_reso_input, self.low_reso_input), 0))
        # high resolution embedding
        self.high_reso_emb3 = self.emb3[:self.high_reso_input.shape[0], ...]
        # low resolution embedding
        self.low_reso_emb3 = self.emb3[self.high_reso_input.shape[0]:, ...]
    
        # prediction
        self.low_low_pred = self.low_reso_pred_layer(self.low_reso_emb3)
        # self.high_low_pred = self.low_reso_pred_layer(self.high_reso_emb3)

        # normalization target (do not use)
        # self.norm_low_reso_target = (torch.argsort(self.low_low_pred, dim = 0) / self.low_low_pred.shape[0]).to(self.device)
        # self.norm_low_reso_target = torch.tensor([0.5]).repeat(self.low_low_pred.shape[0], 1)


    def calculate_losses(self):
        self.low_reso_loss = self.criterion_low_reso(self.low_low_pred, self.low_reso_time.unsqueeze(dim = 1), self.low_reso_status.unsqueeze(dim = 1))
        return self.lambda2 * self.low_reso_loss


class ClassCoxModel(BaseModel):
    def __init__(self, opt):
        super(ClassCoxModel, self).__init__(opt)
        
        # specify the feature extractor structures
        self.feature_extractor_layer = FeatureExtractor(opt).to(self.device)
        # specify neural network structures
        self.discrim_layer = Discriminator(opt).to(self.device)
        # add prediction layers
        self.high_reso_pred_layer = define_linear(opt["feature_dim"], opt["high_reso_output_shape"], activation = None).to(self.device)
        self.low_reso_pred_layer = define_linear(opt["feature_dim"], opt["low_reso_output_shape"], opt["dropout_rate"], nn.Sigmoid()).to(self.device)
        # add all layer names
        self.net_name_list = ["feature_extractor_layer", "discrim_layer", "low_reso_pred_layer", "high_reso_pred_layer"]

        # add loss functions
        self.criterion_high_reso = nn.CrossEntropyLoss().to(self.device)
        if self.loss_type == "LogNeg":
            self.criterion_low_reso = NegativeLogLikelihood(opt).to(self.device)
        elif self.loss_type == "CoxPH":
            self.criterion_low_reso = CoxPH(opt).to(self.device)
        else:
            raise ValueError("Loss function must be CoxPH or LogNeg!")

        # define loss types
        self.loss_names = ["high_reso_loss", "low_reso_loss", "transfer_loss"]
        self.loss_rec = pd.DataFrame(dict(zip(self.loss_names, [[]] * len(self.loss_names))))
        
        # add optimizers
        self.optimizer1 = torch.optim.Adam(list(self.feature_extractor_layer.parameters()) + 
                                           list(self.high_reso_pred_layer.parameters()) + 
                                           list(self.low_reso_pred_layer.parameters()), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        self.optimizer2 = torch.optim.Adam(self.discrim_layer.parameters(), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        
        
        
    def set_input(self, input1, input2):
        self.high_reso_input = Variable(input1["data"].float().to(self.device))
        self.high_reso_label = Variable(input1["label"].long().to(self.device))
        self.low_reso_input = Variable(input2["data"].float().to(self.device))
        self.low_reso_time = Variable(input2["time"].long().to(self.device))
        self.low_reso_status = Variable(input2["status"].float().to(self.device))
        

    def forward(self):
        _, self.emb3 = self.feature_extractor_layer(torch.concat((self.high_reso_input, self.low_reso_input), 0))
        # high resolution embedding
        self.high_reso_emb3 = self.emb3[:self.high_reso_input.shape[0], ...]
        # low resolution embedding
        self.low_reso_emb3 = self.emb3[self.high_reso_input.shape[0]:, ...]
    
        # prediction
        self.high_high_pred = self.high_reso_pred_layer(self.high_reso_emb3)
        self.low_low_pred = self.low_reso_pred_layer(self.low_reso_emb3)
        

    def calculate_losses(self):
        self.high_reso_loss = self.criterion_high_reso(self.high_high_pred, self.high_reso_label)
        self.low_reso_loss = self.criterion_low_reso(self.low_low_pred, self.low_reso_time.unsqueeze(dim = 1), self.low_reso_status.unsqueeze(dim = 1))
        return self.lambda1 * self.high_reso_loss + self.lambda2 * self.low_reso_loss  
        

class BlankClassModel(BaseModel):
    def __init__(self, opt):
        super(BlankClassModel, self).__init__(opt)
        
        # specify the feature extractor structures
        self.feature_extractor_layer = FeatureExtractor(opt).to(self.device)
        # specify neural network structures
        self.discrim_layer = Discriminator(opt).to(self.device)
        # add prediction layers
        self.low_reso_pred_layer = nn.Sequential(define_linear(opt["feature_dim"], opt["low_reso_output_shape"], activation = None)).to(self.device)
        # add all layer names
        self.net_name_list = ["feature_extractor_layer", "discrim_layer", "low_reso_pred_layer"]


        # add loss functions
        self.criterion_low_reso = nn.CrossEntropyLoss().to(self.device)

        # define loss types
        self.loss_names = ["low_reso_loss", "transfer_loss"]
        self.loss_rec = pd.DataFrame(dict(zip(self.loss_names, [[]] * len(self.loss_names))))
        
        # add optimizers
        self.optimizer1 = torch.optim.Adam(list(self.feature_extractor_layer.parameters()) + 
                                           list(self.low_reso_pred_layer.parameters()), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        self.optimizer2 = torch.optim.Adam(self.discrim_layer.parameters(), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        
        
        
    def set_input(self, input1, input2):
        self.high_reso_input = Variable(input1["data"].float().to(self.device))
        self.low_reso_input = Variable(input2["data"].float().to(self.device))
        self.low_reso_label = Variable(input2["label"].long().to(self.device))


    def forward(self):
        _, self.emb3 = self.feature_extractor_layer(torch.concat((self.high_reso_input, self.low_reso_input), 0))
        # high resolution embedding
        self.high_reso_emb3 = self.emb3[:self.high_reso_input.shape[0], ...]
        # low resolution embedding
        self.low_reso_emb3 = self.emb3[self.high_reso_input.shape[0]:, ...]
        # prediction
        self.low_low_pred = self.low_reso_pred_layer(self.low_reso_emb3)

    def calculate_losses(self):
        # low resolution loss
        self.low_reso_loss = self.criterion_low_reso(self.low_low_pred, self.low_reso_label)

        return self.lambda2 * self.low_reso_loss 
    

class ClassClassModel(BaseModel):
    def __init__(self, opt):
        super(ClassClassModel, self).__init__(opt)
        
        # specify the feature extractor structures
        self.feature_extractor_layer = FeatureExtractor(opt).to(self.device)
        # specify neural network structures
        self.discrim_layer = Discriminator(opt).to(self.device)
        # add prediction layers
        self.high_reso_pred_layer = nn.Sequential(define_linear(opt["feature_dim"], opt["high_reso_output_shape"], activation = None)).to(self.device)
        self.low_reso_pred_layer = nn.Sequential(define_linear(opt["feature_dim"], opt["low_reso_output_shape"], activation = None)).to(self.device)
        # add all layer names
        self.net_name_list = ["feature_extractor_layer", "discrim_layer", "high_reso_pred_layer", "low_reso_pred_layer"]


        # add loss functions
        self.criterion_high_reso = nn.CrossEntropyLoss().to(self.device)
        self.criterion_low_reso = nn.CrossEntropyLoss().to(self.device)

        # define loss types
        self.loss_names = ["high_reso_loss", "low_reso_loss", "transfer_loss"]
        self.loss_rec = pd.DataFrame(dict(zip(self.loss_names, [[]] * len(self.loss_names))))
        
        # add optimizers
        self.optimizer1 = torch.optim.Adam(list(self.feature_extractor_layer.parameters()) + 
                                           list(self.high_reso_pred_layer.parameters()) + 
                                           list(self.low_reso_pred_layer.parameters()), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        self.optimizer2 = torch.optim.Adam(self.discrim_layer.parameters(), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        
        
        
    def set_input(self, input1, input2):
        self.high_reso_input = Variable(input1["data"].float().to(self.device))
        self.high_reso_label = Variable(input1["label"].long().to(self.device))
        self.low_reso_input = Variable(input2["data"].float().to(self.device))
        self.low_reso_label = Variable(input2["label"].long().to(self.device))


    def forward(self):
        _, self.emb3 = self.feature_extractor_layer(torch.concat((self.high_reso_input, self.low_reso_input), 0))
        # high resolution embedding
        self.high_reso_emb3 = self.emb3[:self.high_reso_input.shape[0], ...]
        # low resolution embedding
        self.low_reso_emb3 = self.emb3[self.high_reso_input.shape[0]:, ...]
        # prediction
        self.high_high_pred = self.high_reso_pred_layer(self.high_reso_emb3)
        self.low_low_pred = self.low_reso_pred_layer(self.low_reso_emb3)

    def calculate_losses(self):
        # high resolution loss
        self.high_reso_loss = self.criterion_high_reso(self.high_high_pred, self.high_reso_label)
        # low resolution loss
        self.low_reso_loss = self.criterion_low_reso(self.low_low_pred, self.low_reso_label)

        return self.lambda1 * self.high_reso_loss + self.lambda2 * self.low_reso_loss 
    

class BlankBCEModel(BaseModel):
    def __init__(self, opt):
        super(BlankBCEModel, self).__init__(opt)
        
        # specify the feature extractor structures
        self.feature_extractor_layer = FeatureExtractor(opt).to(self.device)
        # specify neural network structures
        self.discrim_layer = Discriminator(opt).to(self.device)
        # add prediction layers
        self.low_reso_pred_layer = nn.Sequential(define_linear(opt["feature_dim"], opt["low_reso_output_shape"], activation = None)).to(self.device)
        # add all layer names
        self.net_name_list = ["feature_extractor_layer", "discrim_layer", "low_reso_pred_layer"]

        # add loss functions
        self.criterion_low_reso = BCELoss(opt).to(self.device)

        # define loss types
        self.loss_names = ["low_reso_loss", "transfer_loss"]
        self.loss_rec = pd.DataFrame(dict(zip(self.loss_names, [[]] * len(self.loss_names))))
        
        # add optimizers
        self.optimizer1 = torch.optim.Adam(list(self.feature_extractor_layer.parameters()) + 
                                           list(self.low_reso_pred_layer.parameters()), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        self.optimizer2 = torch.optim.Adam(self.discrim_layer.parameters(), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        
        
        
    def set_input(self, input1, input2):
        self.high_reso_input = Variable(input1["data"].float().to(self.device))
        self.low_reso_input = Variable(input2["data"].float().to(self.device))
        self.low_reso_time = Variable(input2["time"].float().to(self.device))
        self.low_reso_status = Variable(input2["status"].float().to(self.device))
        

    def forward(self):
        _, self.emb3 = self.feature_extractor_layer(torch.concat((self.high_reso_input, self.low_reso_input), 0))
        # high resolution embedding
        self.high_reso_emb3 = self.emb3[:self.high_reso_input.shape[0], ...]
        # low resolution embedding
        self.low_reso_emb3 = self.emb3[self.high_reso_input.shape[0]:, ...]
        # prediction
        self.low_low_pred = self.low_reso_pred_layer(self.low_reso_emb3)

    def calculate_losses(self):
        self.low_reso_loss = self.criterion_low_reso(self.low_low_pred, self.low_reso_time.unsqueeze(dim = 1), self.low_reso_status.unsqueeze(dim = 1))
        return self.lambda2 * self.low_reso_loss
    
class ClassBCEModel(BaseModel):
    def __init__(self, opt):
        super(ClassBCEModel, self).__init__(opt)
        
        # specify the feature extractor structures
        self.feature_extractor_layer = FeatureExtractor(opt).to(self.device)
        # specify neural network structures
        self.discrim_layer = Discriminator(opt).to(self.device)
        # add prediction layers
        self.high_reso_pred_layer = define_linear(opt["feature_dim"], opt["high_reso_output_shape"], activation = None).to(self.device)
        self.low_reso_pred_layer = nn.Sequential(define_linear(opt["feature_dim"], opt["low_reso_output_shape"], activation = None)).to(self.device)
        # add all layer names
        self.net_name_list = ["feature_extractor_layer", "discrim_layer", "low_reso_pred_layer", "high_reso_pred_layer"]

        # add loss functions
        self.criterion_high_reso = nn.CrossEntropyLoss().to(self.device)
        self.criterion_low_reso = BCELoss(opt).to(self.device)

        # define loss types
        self.loss_names = ["high_reso_loss", "low_reso_loss", "transfer_loss"]
        self.loss_rec = pd.DataFrame(dict(zip(self.loss_names, [[]] * len(self.loss_names))))
        
        # add optimizers
        self.optimizer1 = torch.optim.Adam(list(self.feature_extractor_layer.parameters()) + 
                                           list(self.high_reso_pred_layer.parameters()) + 
                                           list(self.low_reso_pred_layer.parameters()), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        self.optimizer2 = torch.optim.Adam(self.discrim_layer.parameters(), 
                                           lr = opt["lr"], betas = (opt["beta1"], 0.999))
        
        
        
    def set_input(self, input1, input2):
        self.high_reso_input = Variable(input1["data"].float().to(self.device))
        self.high_reso_label = Variable(input1["label"].long().to(self.device))
        self.low_reso_input = Variable(input2["data"].float().to(self.device))
        self.low_reso_time = Variable(input2["time"].float().to(self.device))
        self.low_reso_status = Variable(input2["status"].float().to(self.device))
        

    def forward(self):
        _, self.emb3 = self.feature_extractor_layer(torch.concat((self.high_reso_input, self.low_reso_input), 0))
        # high resolution embedding
        self.high_reso_emb3 = self.emb3[:self.high_reso_input.shape[0], ...]
        # low resolution embedding
        self.low_reso_emb3 = self.emb3[self.high_reso_input.shape[0]:, ...]
    
        # prediction
        self.high_high_pred = self.high_reso_pred_layer(self.high_reso_emb3)
        self.low_low_pred = self.low_reso_pred_layer(self.low_reso_emb3)
        

    def calculate_losses(self):
        self.high_reso_loss = self.criterion_high_reso(self.high_high_pred, self.high_reso_label)
        self.low_reso_loss = self.criterion_low_reso(self.low_low_pred, self.low_reso_time.unsqueeze(dim = 1), self.low_reso_status.unsqueeze(dim = 1))
        return self.lambda1 * self.high_reso_loss + self.lambda2 * self.low_reso_loss  
        






































