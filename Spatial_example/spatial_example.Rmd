---
title: "DEGAS_erickson"
output: html_document
date: "2025-05-10"
---


```{r}
library(ggplot2)
library(reticulate)
library(dplyr)
library(magrittr)
library(Seurat)
library(stringr)
library(DESeq2)
library(DEGASv2)
library(ggpubr)
library(pROC)
library(PRROC)
```


Run the Erickson 1 dataset
create Seurat Object for each ST slides

```{r}
root_dir <- "/N/u/lisih/Quartz/Desktop/erickson_patient1"
```

```{r}
folder_names <- list.dirs(paste0(root_dir, "/h5_files"))
for (st_name in folder_names[2:length(folder_names)]) {
  st_name <- strsplit(st_name, "/")[[1]]
  st_name <- st_name[length(st_name)]
  print(st_name)
  # load h5 file
  h5_data <- Read10X_h5(paste0(root_dir, "/h5_files/", st_name, "/filtered_feature_bc_matrix.h5"))
  spatial_loc <- read.csv(paste0(root_dir, "/h5_files/", st_name, "/tissue_positions_list.csv"), header = FALSE)[, c(1, 5, 6)]
  rownames(spatial_loc) <- spatial_loc$V1
  spatial_loc <- spatial_loc[, c(2, 3)]
  barcodes <- intersect(colnames(h5_data), rownames(spatial_loc))
  h5_data <- h5_data[, barcodes]
  spatial_loc <- spatial_loc[barcodes, ]
  colnames(spatial_loc) <- c("coord1", "coord2")
  
  createSTobj(h5_data, spatial_loc, st_name, "erickson_patient1") 
}
```

Load all ST and patient datasets
```{r}
load(paste0(root_dir, "/Pat_raw_counts/PatDat.RData"))
st_list <- list()
diff_expr_files <- list()
folder_names <- list.dirs("erickson_patient1/ST_raw_counts")
for (st_name in folder_names[2:length(folder_names)]){
  st_name <- strsplit(st_name, "/")[[1]]
  st_name <- st_name[length(st_name)]
  print(st_name)
  st_list <- append(st_list, readRDS(paste0(root_dir, "/ST_raw_counts/", st_name, "/", st_name, ".rds")))
  diff_expr_files <- append(diff_expr_files, paste0(root_dir, "/ST_raw_counts/", st_name, "/st_diff_expr.csv"))
}
```

DEGAS Preprocessing all

```{r}
DEGAS_preprocessing(bulk_dataset = bulk_dataset, phenotype = phenotype, st_list = st_list, results_dir = "erickson_patient1",
                    st_lab_list = c(), model_type = "survival", diff_expr_files = diff_expr_files)
```

Now let's visualize erickson1 dataset

```{r}
plot_erickson1 <- function(folder_path) {
    result_files <- list.files(folder_path, pattern = "\\.csv$", ignore.case = TRUE, full.names = TRUE)
    result_dfs <- data.frame()
    for (st_res in result_files) {
      st_name <- strsplit(st_res, "/")[[1]]
      st_name <- strsplit(st_name[length(st_name)], ".csv")[[1]]
      if ((st_name %in% c("summary_mean", "summary")) == FALSE) {
        # print(st_name)
        result_df <- read.csv2(st_res)
        result_df["sample"] <- st_name
        result_dfs <- rbind(result_dfs, result_df)
      }
    }

    tissue_type <- read.csv("/N/u/lisih/Quartz/Desktop/erickson_patient1/histology_and_benign_gland_risk_groups_coordinates.csv")
    # Merge the dataframes
    merge_result_dfs <- merge(result_dfs, tissue_type, by.x = c("x", "y", "sample"), by.y = c("coord1", "coord2", "sample"), all = FALSE)
    merge_result_dfs <- merge_result_dfs %>%
      mutate(group = case_when(
        group %in% c("BG_rank1", "BG_rank2", "BG_rank3", "BG_rank4") ~ "BG",
        TRUE ~ as.character(group)
      ))
    merge_result_dfs <- merge_result_dfs[merge_result_dfs$group != "", ]
    
    merge_result_dfs2 <- merge_result_dfs %>%
      mutate(group = case_when(
        ! group %in% c("GG1", "GG2", "GG4", "GG4 Cribriform") ~ "Non-Tumor",
        TRUE ~ as.character(group)
      ))
    merge_result_dfs3 <- merge_result_dfs %>%
      mutate(group = case_when(
        group %in% c("GG1", "GG2", "GG4", "GG4 Cribriform") ~ "Tumor",
        TRUE ~ as.character(group)
      ))
    merge_result_dfs4 <- merge_result_dfs3 %>%
      mutate(group = case_when(
        !group %in% c("Tumor") ~ "Non-Tumor",
        TRUE ~ as.character(group)
      ))
    
    # make boxplot
    x_axis_order = c("Non-Tumor", "GG1", "GG2", "GG4", "GG4 Cribriform")
    colors <-  c("Non-Tumor" = "gray", "GG1" = "#FEFF91", 
                      "GG2" = "#E69F00", "GG4" = "#A73030FF", "GG4 Cribriform" = "darkred")
    my_comparisons <- list(c("Non-Tumor", "GG1"), c("Non-Tumor", "GG2"), c("Non-Tumor", "GG4"), c("Non-Tumor", "GG4 Cribriform"))
    p2 <- ggplot(merge_result_dfs2, aes(x = factor(group, levels = x_axis_order), y = hazard)) +
      geom_boxplot(aes(fill = group), alpha = 0.5) +
      # geom_jitter(width = 0.2, alpha = 0.5, color = "gray") + 
      labs(title = "Tumor versus Non-Tumor",
           x = "Tissue Type",
           y = "Hazard Score",
           fill = "Ground Truth") +
      theme(
          axis.text.x = element_text(angle = 0, hjust = 0.5, face = "bold"),
          text = element_text(face = "bold"),
          plot.subtitle = element_text(face = "bold"),
          plot.caption = element_text(face = "bold")
        ) + 
      scale_fill_manual(values = colors) + 
      stat_compare_means(comparisons = my_comparisons, size = 5)
    ggsave(filename = paste0(folder_path, "/", "box.pdf"), plot = p2, width = 7, height = 7, units = "in") 
    
    # Calculate ROC curve and AUC
    merge_result_dfs2$binary <- as.integer(merge_result_dfs2$group %in% c("GG1", "GG2", "GG4", "GG4 Cribriform"))
    roc_data <- roc(merge_result_dfs2$binary, merge_result_dfs2$hazard)
    roc_df <- data.frame(sensitivity = roc_data$sensitivities, specificity = roc_data$specificities)
    # Plot AUC-ROC curve using ggplot2
    roc_degas <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity)) +
      geom_line(color = "darkred") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
      labs(x = "False Positive Rate", y = "True Positive Rate") + 
      geom_text(aes(label = paste0("AUC-ROC: ", round(roc_data$auc, 4)), x = 0.5, y = 0.5),
                   color = "black", size = 10) + 
      theme(
          axis.text.x = element_text(face = "bold"),
          text = element_text(face = "bold"),
          plot.subtitle = element_text(face = "bold"),
          plot.caption = element_text(face = "bold")
        ) 
    
    ggsave(filename = paste0(folder_path, "/", "auc_roc.pdf"), plot = roc_degas, width = 7, height = 7, units = "in") 
    
    # make zoom out plot
    x_axis_order = c("BG", "Exclude", "Fat", "PIN", "Stroma", "Nerve", "Chronic inflammation", "Inflammation", "Transition_State", "Vessel", "Tumor")
    colors <- c("BG" = "gray", "Exclude" = "gray", "Fat" = "gray", "PIN" = "gray", "Stroma" = "gray", "Nerve" = "#E69F00", "Chronic inflammation" = "#E69F00", "Inflammation" = "#E69F00", "Transition_State" = "#E69F00", "Vessel" = "#E69F00", "Tumor"= "darkred")
    p3 <- ggplot(merge_result_dfs3, aes(x = factor(group, levels = x_axis_order), y = hazard)) +
      geom_boxplot(aes(fill = group), alpha = 0.5) +
      # geom_jitter(width = 0.2, alpha = 0.5, color = "gray") + 
      labs(title = "Zoom Out Non-Tumor Regions",
           x = "Tissue Type",
           y = "Hazard Score",
           fill = "Ground Truth") +
      theme(
          axis.text.x = element_text(angle = 30, hjust = 1, face = "bold"),
          text = element_text(face = "bold"),
          plot.subtitle = element_text(face = "bold"),
          plot.caption = element_text(face = "bold")
        ) +
      scale_fill_manual(values = colors)
    ggsave(filename = paste0(folder_path, "/", "box2.pdf"), plot = p3, width = 10, height = 7, units = "in")
    
    # remake the spatial visualization 
    for (st_res in result_files) {
      st_name <- strsplit(st_res, "/")[[1]]
      st_name <- strsplit(st_name[length(st_name)], ".csv")[[1]]
      if ((st_name %in% c("summary_mean", "summary")) == FALSE) {
        # print(st_name)
        sub_results <- merge_result_dfs4[merge_result_dfs4$sample == st_name, ]
        percentage_of_Tumor <- sum(sub_results$group == "Tumor") / nrow(sub_results) * 100
        p <- ggplot(sub_results, aes(x = x, y = y, color = hazard, shape = group)) +
          geom_point() +
          scale_color_gradient(low = "gray", high = "darkred", limits = c(0.0, 1.0)) +
          scale_shape_manual(values = c("Tumor" = 16, "Non-Tumor" = 4)) +
          labs(title = paste0(st_name, ": Tumor Perc ", round(percentage_of_Tumor, 2), "%"), shape = "Ground Truth", color = "Hazard") + 
          theme(
            title = element_text(face = "bold", size = 15),
            legend.title = element_text(face = "bold"),
            legend.text = element_text(face = "bold"),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.x=element_blank(),
            axis.ticks.y=element_blank() 
          )

      ggsave(filename = paste0(folder_path, "/", st_name, "_scatter.pdf"), plot = p, width = 5.5, height = 5, units = "in")
      }
    }
    
    return(roc_data)
}
```


Load python env and run the model

```{r}
results_df <- data.frame()
for (data_type in c("SDE")) {
  for (loss_type in c("rank_loss")) {
    for (transfer_type in c("Wasserstein")) {
      data_dir <- paste0("/N/u/lisih/Quartz/Desktop/erickson_patient1/", data_type, ".RData")
      model_save_dir <- paste0("/N/u/lisih/Quartz/Desktop/checkpoints/erickson_patient1_",
                  data_type, "_", transfer_type, "_", loss_type, "_BlankCox_", Sys.Date())
      run_DEGAS_SCST(DEGAS_data_file = data_dir, model_type = "BlankCox", data_name = "erickson_patient1",
                     loss_type = loss_type, transfer_type = transfer_type, model_save_dir = model_save_dir, tot_iters = 300)
      roc_data <- plot_erickson1(folder_path = model_save_dir)
      results_df %<>% rbind(data.frame(data_name = "erickson1", data_type = data_type, loss_type = loss_type, transfer_type = transfer_type,
                                 auc = round(roc_data$auc, 4)))
    }
  }
}
```



