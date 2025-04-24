from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from .tools import *

class STSCDataset(Dataset):
    # load spatial transcriptomics datasets
    # the logic of this Dataset is different from the pytorch
    # in this STDataset, each batch samples is saved as a single sample in pytorch Dataset
    # which means, each sample is batch_size X feature_dim
    def __init__(self, st_expr_mat, sc_lab_mat = None, 
                random_seed = 0, tot_iters = 300, batch_size = 200, phase = "train", sample_method = "balance"):
        """
        random_seed (int): random seed for shuffling the dataset
        sample_method: which method to sample
        tot_iters: total number of training iteration
        batch_size (int): batch size
        phase (string): should be "train" or "eval".
        """
        super(STSCDataset, self).__init__()
        self.n_samples = st_expr_mat.shape[0]
        # if st_loc_mat is not None:
        #     assert self.n_samples == st_loc_mat.shape[0]
        #     self.stLoc = st_loc_mat
        # else:
        #     self.stLoc = np.zeros((self.n_samples, 2)) # dummy value
        if sc_lab_mat is not None:
            if len(sc_lab_mat.shape) > 1:
                if sc_lab_mat.shape[1] > 1:
                    sc_lab_mat = np.argmax(sc_lab_mat, axis = 1)
                sc_lab_mat = sc_lab_mat.flatten()
            assert self.n_samples == sc_lab_mat.shape[0]
            self.scLab = sc_lab_mat
        else:
            self.scLab = np.zeros((self.n_samples)).astype(np.int8) # dummy value

        self.stDat = st_expr_mat
        self.st_idx = np.array(range(self.n_samples))

        self.phase = phase
        self.tot_iters = tot_iters
        self.random_seed = random_seed
        self.sample_method  = sample_method
        self.batch_size = batch_size


    def __len__(self):
        if self.phase == "train":
            return self.tot_iters
        else:
            return self.n_samples
        
    def get(self, index):
        assert self.phase == "train"
        if self.sample_method == "balance":
            self.idx_select = balance_sampling(self.scLab, self.batch_size, self.random_seed + index)
        else:
            raise ValueError("Other sampling method is not implemented yet!")

    def __getitem__(self, index):
        if self.phase == "train":
            self.get(index)
        else:
            self.idx_select = index

        return {"index": np.array(self.idx_select), "data": self.stDat[self.idx_select, :], "label": self.scLab[self.idx_select]}
#####################################################################################

class PatDataset(Dataset):
    def __init__(self, pat_expr_mat, pat_lab_mat, random_seed = 0, batch_size = 200, tot_iters = 300, phase = "train", model_type = "phenotype", sample_method = "balance"):
        super(PatDataset, self).__init__()

        self.patDat = pat_expr_mat
        self.n_samples = pat_expr_mat.shape[0]
        assert self.n_samples == pat_lab_mat.shape[0]
        if (model_type in ["BlankClass", "ClassClass"]):
            if len(pat_lab_mat.shape) > 1:
                if (pat_lab_mat.shape[1] > 1):
                    pat_lab_mat = np.argmax(np.array(pat_lab_mat), axis = 1)
                pat_lab_mat = pat_lab_mat.flatten()
            self.patLab = pat_lab_mat
        else:
            self.time = np.array(pat_lab_mat[:, 0]).astype(int).flatten()
            self.status = np.array(pat_lab_mat[:, 1]).flatten()
        self.pat_idx = np.array(range(self.n_samples))

        self.random_seed = random_seed
        self.batch_size = batch_size
        self.tot_iters = tot_iters
        self.phase = phase
        self.model_type = model_type
        self.sample_method = sample_method
    


    def __len__(self):
        if self.phase == "train":
            return self.tot_iters
        else:
            return self.n_samples

    def get(self, index):
        assert self.phase == "train"
        if not (self.model_type in ["BlankClass", "ClassClass"]):
            np.random.seed(self.random_seed + index)
            self.idx_select = np.random.choice(self.pat_idx, self.batch_size, replace = False)
        elif (self.sample_method == "balance"):
            self.idx_select = balance_sampling(self.patLab, self.batch_size, self.random_seed + index)
        else:
            raise ValueError("Other sampling method is not implemented yet!")

        
    def __getitem__(self, index):
        if self.phase == "train":
            self.get(index)
        else:
            self.idx_select = index
        if not (self.model_type in ["BlankClass", "ClassClass"]):
            return {"pid": self.pat_idx[self.idx_select], "data": self.patDat[self.idx_select, :], "time": self.time[self.idx_select], "status": self.status[self.idx_select]}
        else:
            return {"pid": self.pat_idx[self.idx_select], "data": self.patDat[self.idx_select, :], "label": self.patLab[self.idx_select]}





