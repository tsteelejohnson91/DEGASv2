import os
import pandas as pd
import numpy as np
from glob import glob
from .models import *
from .datasets import load_datasets
import time
from tqdm import tqdm
from .tools import EarlyStopper
import seaborn as sns
import matplotlib.pyplot as plt

# for time discretization
from pycox.models import LogisticHazard

# random seed for reproducibility
import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def run_model(opt, pat_expr_mat, pat_lab_mat, sc_expr_mat, sc_lab_mat = None):
    early_stop_flag = False
    # split train and eval if early stopping
    if opt["early_stopping"]:
        # not run now
        opt["sc_batch_size"] = int(opt["sc_batch_size"] * 1.2) # inflate 20% for validation early stopping
        opt["pat_batch_size"] = int(opt["pat_batch_size"] * 1.2) # inflate 20% for validation early stopping
        early_stop_loss_rec = pd.DataFrame({})
        early_stopper = EarlyStopper(patience = opt["patience"], min_delta = opt["min_delta"], epsilon = opt["epsilon"])

    if "BCE" in opt["model_type"]:
        time, status = pat_lab_mat[:, 0].squeeze(), pat_lab_mat[:, 1].squeeze()
        if len(np.unique(status)) > 2:
            time, status = status, time
        labtrans = LogisticHazard.label_transform(opt["low_reso_output_shape"]) # how many time windows do we use
        idx_time, event = labtrans.fit_transform(*(time.astype("float32"), status.astype("float32")))
        pat_lab_mat = np.vstack([idx_time, event]).T

    # define data loaders and model # modity dataset
    high_reso_loader, low_reso_loader = load_datasets("train", opt, pat_expr_mat, pat_lab_mat, sc_expr_mat, sc_lab_mat)
    high_reso_eval_loader, low_reso_eval_loader = load_datasets("test", opt, pat_expr_mat, pat_lab_mat, sc_expr_mat, sc_lab_mat)

    # define the model
    first_item = next(iter(low_reso_eval_loader))
    opt["input_shape"] = first_item["data"].shape[1]
    # check optionals for the model
    if opt["model_type"] in ["ClassClass", "ClassCox", "ClassBCE"]:
        opt["high_reso_output_shape"] = len(np.unique(sc_lab_mat.squeeze()))
    model = load_models(opt)

    # train and evaluate the model
    for epoch, (high_reso_data, low_reso_data) in enumerate(tqdm(zip(high_reso_loader, low_reso_loader)), start = 1):
        # print("Training Epoch {}".format(epoch))
        high_reso_data = {key: value.squeeze(0) for key, value in high_reso_data.items()}
        low_reso_data = {key: value.squeeze(0) for key, value in low_reso_data.items()}

        if opt["early_stopping"]:
            # should permuate the dataset to avoid sc label as 0, 0, 0, 1, 1, 1, 2, 2, 2, ...
            np.random.seed(opt["seed"])
            high_reso_size = int(high_reso_data["data"].shape[0])
            low_reso_size = int(low_reso_data["data"].shape[0])
            permuate_high_reso_index = np.random.choice(range(high_reso_size), high_reso_size, replace = False)
            permuate_low_reso_index = np.random.choice(range(low_reso_size), low_reso_size, replace = False)
            high_reso_val_data = {key: value[permuate_high_reso_index[int(high_reso_size * 0.8):], ...] for key, value in high_reso_data.items()}
            high_reso_data = {key: value[permuate_high_reso_index[:int(high_reso_size * 0.8)], ...] for key, value in high_reso_data.items()}
            low_reso_val_data = {key: value[permuate_low_reso_index[int(low_reso_size * 0.8):], ...] for key, value in low_reso_data.items()}
            low_reso_data = {key: value[permuate_low_reso_index[:int(low_reso_size * 0.8)], ...] for key, value in low_reso_data.items()}


        model.set_input(high_reso_data, low_reso_data) # notice that when training, the data shape will be 1 X batch_size X feature size, we will squeeze it in our backend code
        model.optimize_parameters(epoch)

        # each if early stopping
        if opt["early_stopping"]:
            model.set_evaluate_mode()
            model.set_input(high_reso_val_data, low_reso_val_data)
            model.forward()
            early_stop_loss_rec = pd.concat([early_stop_loss_rec, model.record_early_stopping_losses()], axis = 0)
            early_stop_flag = early_stopper.early_stop(early_stop_loss_rec)
            model.set_train_mode()
    
        if ((opt["is_save"]) and (epoch % opt["save_freq"] == 0)) or (epoch == opt["tot_iters"]) or early_stop_flag:    
            print("Saving the model...")
            model.save_networks(epoch)
            
            # print("Evaluate the model...")
            model.set_evaluate_mode()
            high_reso_results, high_reso_embs = model.linear_eval(high_reso_eval_loader, opt["extract_embs"])
            low_reso_results, low_reso_embs = model.linear_eval(low_reso_eval_loader, opt["extract_embs"])

            np.save(os.path.join(model.save_dir, "high_reso_embs_epoch_{}.npy".format(epoch)), high_reso_embs)
            np.save(os.path.join(model.save_dir, "low_reso_embs_epoch_{}.npy".format(epoch)), low_reso_embs)
            high_reso_results.to_csv(os.path.join(model.save_dir, "high_reso_results_epoch_{}.csv".format(epoch)))
            low_reso_results.to_csv(os.path.join(model.save_dir, "low_reso_results_epoch_{}.csv".format(epoch)))
            # print("Back to Training phase")
            model.load_networks(epoch)
            model.set_train_mode()  

            if early_stop_flag:
                break
    
    # save loss functions
    loss_rec = model.loss_rec
    loss_rec.to_csv(os.path.join(model.save_dir, "losses.csv".format(epoch)))
    loss_rec["Epoch"] = list(range(len(loss_rec)))
    loss_rec = loss_rec.melt("Epoch", var_name = "Loss Type", value_name = "Value")
    sns.lineplot(data = loss_rec, x = "Epoch", y = "Value", hue = "Loss Type")
    plt.savefig(os.path.join(model.save_dir, "losses.png".format(epoch)))
    plt.close()
    return model.save_dir, epoch


def bagging_all_results(opt, pat_expr_mat, pat_lab_mat, sc_expr_mat, sc_lab_mat = None):
    """
    sc_expr_mat: single cell or spatial transcriptomic data gene expression
    sc_loc_mat: spatial transcriptomic data (optional, for graph NN)
    sc_lab_mat: single cell labels (optional)
    pat_expr_mat: patient gene expression value
    pat_lab_mat: patient labels
    """
    results = []
    for seed in range(opt["tot_seeds"]):
        opt["seed"] = seed
        print("Run submodel {}...".format(seed))
        if "random_feat" in opt.keys() and opt["random_feat"] and "random_perc" in opt.keys():
            np.random.seed(opt["seed"])
            num_select_feats = np.floor(pat_expr_mat.shape[1] * opt["random_perc"]).astype(int)
            select_feats = np.sort(np.random.choice(list(range(pat_expr_mat.shape[1])), num_select_feats, replace = False))
            save_results_folder, epoch = run_model(opt.copy(), pat_expr_mat[:, select_feats], pat_lab_mat, sc_expr_mat[:, select_feats], sc_lab_mat)
        else:
            save_results_folder, epoch = run_model(opt.copy(), pat_expr_mat, pat_lab_mat, sc_expr_mat, sc_lab_mat)
        result_df = pd.read_csv(os.path.join(save_results_folder, "high_reso_results_epoch_{}.csv".format(epoch)))
        result_df["seed"] = seed
        if "hazard" not in result_df.columns: # multicolumn hazard:
            sc_index = result_df["index"]; result_df.drop("index", axis = 1)
            cum_pmf = np.cumsum(np.array(result_df), axis = 1)
            n_bins = cum_pmf.shape[1]
            indices = (cum_pmf > opt["cum_thresh"]).argmax(axis = 1) # first indices where cum sum larger then cum_thresh
            hazard = 1 - indices / n_bins
            result_df = pd.DataFrame({"index": sc_index, "hazard": hazard})
        results.append(result_df)

    results = pd.concat(results, ignore_index = True)
    results.to_csv(os.path.join(save_results_folder, "summary.csv"))
    results_mean = results.groupby("index").mean().reset_index()
    results_mean.to_csv(os.path.join(save_results_folder, "summary_mean.csv"))
    return results_mean


    
    
