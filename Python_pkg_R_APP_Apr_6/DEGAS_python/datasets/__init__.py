from .dataset import *
# from .graph_dataset import *
# from torch_geometric.loader import DataLoader as GDataLoader
from torch.utils.data import DataLoader

def load_datasets(phase, opt, pat_expr_mat, pat_lab_mat, st_expr_mat, sc_lab_mat = None):
    bs = 1 if phase == "train" else opt["sc_batch_size"]
    # load single cell or st dataset
    high_reso_dataset = DataLoader(STSCDataset(st_expr_mat, sc_lab_mat,
            random_seed = opt["seed"], tot_iters = opt["tot_iters"], batch_size = opt["sc_batch_size"], phase = phase, sample_method = opt["sample_method"]),
            batch_size = bs, shuffle = False)
    low_reso_dataset = DataLoader(PatDataset(pat_expr_mat, pat_lab_mat, random_seed = opt["seed"], batch_size = opt["pat_batch_size"], 
            tot_iters = opt["tot_iters"], phase = phase, model_type = opt["model_type"], sample_method = opt["sample_method"]),
            batch_size = bs, shuffle = False)
 
    return high_reso_dataset, low_reso_dataset



