# set up configurations, we may copy these configurations into a single file
# if total folds = 1, we set fold = -1 which indicate we doesn't split subfolds

Common_opt = {"gpu_id": 0, "feature_dim": 25,  
        "dropout_rate": 0.5, "lr": 0.01, "beta1": 0.9, "save_freq": 50, # parameters do not change
        "transfer_type": "Wasserstein", 
        "lambda2": 1.0, "lambda2": 3.0, "lambda3": 3.0, "gp_weight": 10.0, "n_critic": 2, "tot_iters": 300, # model hyper-parameter information (to change)
        "data_name": "my_data", "sc_batch_size": 200, "pat_batch_size": 200, "sample_method": "balance",
        "is_save": False, "extract_embs": False, "seed": 0, "tot_seeds": 10, "early_stopping": False,
        "patience": 10, "min_delta": 0.01, "epsilon": 0.05
        }


BlankClass_opt = {**Common_opt, **{"low_reso_output_shape": 2, "model_type": "BlankClass", "loss_type": "cross_entropy"}}

ClassClass_opt = {**Common_opt, **{"low_reso_output_shape": 2, "model_type": "ClassClass", "loss_type": "cross_entropy"}}

BlankCox_opt = {**Common_opt, **{"low_reso_output_shape": 1, "model_type": "BlankCox", "loss_type": "CoxPH"}}

ClassCox_opt = {**Common_opt, **{"low_reso_output_shape": 1, "model_type": "ClassCox", "loss_type": "CoxPH"}}

BlankBCE_opt = {**Common_opt, 
                  **{"model_type": "BlankBCE",  "low_reso_output_shape": 10, "loss_type": "BCE", "lambda2": 1e-1, "cum_thresh": 0.5, "tot_iters": 100}}

ClassBCE_opt = {**Common_opt, 
                  **{"model_type": "ClassBCE",  "low_reso_output_shape": 10, "loss_type": "BCE", "lambda1": 1.0, "lambda2": 1e-1, "cum_thresh": 0.5, "tot_iters": 100}}





