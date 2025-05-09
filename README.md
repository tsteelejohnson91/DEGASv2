## Diagnostic Evidence GAuge of Single cells (DEGAS) version 2

**Package development by:**

**Ziyu Liu**

**Travis S. Johnson (https://github.com/tsteelejohnson91)**

**Sihong Li (https://github.com/alanli97)**
![DEGASv2 Figure 1-2](figures/DEGASv2_fig1-2.png)
DEGASv2

## **Installation**

Step 1 Install devtools in R

```{r}
install.packages("devtools")
```

Step 2 Install Biocmanager in R

```{r}
install.packages("BiocManager")
```

Step 3 Install DESeq2

```{r}
library(BiocManager)
BiocManager::install("DESeq2")
```

Step 4 Install DEGASv2 in R

```{r}
install_github("alanli97/DEGASv2")
```

## **Prerequisites**

**OS**

OSX

Linux

**Python packages**

tensorflow

functools

numpy

math

**R**

Rtsne

ggplot2

DESeq2
