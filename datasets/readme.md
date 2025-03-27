# scSGC
scSGC, a Soft Graph Clustering for single-cell RNA sequencing data, aims to more accurately characterize continuous similarities among cells through non-binary edge weights, thereby mitigating the limitations of rigid data structures. 
See details in our paper: "Soft Graph Clustering for single-cell RNA Sequencing Data" published in XXX（CCF-X）.
（Accepted as a long paper for the research track at DASFAA 2024）

(arXiv: XXX)

（DOI：XXX ）

# Background
Clustering analysis is fundamental in single-cell RNA sequencing (scRNA-seq) data analysis for elucidating cellular heterogeneity and diversity. 
Recent graph-based scRNA-seq clustering methods, particularly graph neural networks (GNNs), have significantly improved in tackling the challenges of high-dimension, high-sparsity, and frequent dropout events that lead to ambiguous cell population boundaries. 
However, the GNN-based method is intended for general graph encoding and faces challenges when applied to scRNA-seq data due to the following reasons: 
(i) GNN-based methods typically construct hard graphs from similarity matrices by applying a threshold that overly simplifies intercellular relationships into binary edges (0 or 1), which restricts the capture of continuous similarity features among cells and leads to significant information loss. 
(ii) Hard graphs derived from scRNA-seq data, which often exhibit significant inter-cluster connections, present challenges for GNN-based methods that rely heavily on underlying graph structures and typically generate similar representations for neighboring nodes, thereby leading to erroneous message propagation and biased clustering outcomes. 


# Overview
To tackle these challenges, we introduce scSGC, a Soft Graph Clustering for single-cell RNA sequencing data, which aims to more accurately characterize continuous similarities among cells through non-binary edge weights, thereby mitigating the limitations of rigid data structures. 
The scSGC framework comprises three core components: 
(i) a zero-inflated negative binomial (ZINB)-based feature autoencoder designed to effectively handle the sparsity and dropout issues in scRNA-seq data; 
(ii) a dual-channel cut-informed soft graph embedding module, constructed through deep graph-cut information, which captures continuous similarities between cells while simultaneously preserving scRNA-seq data structures; 
and (iii) an optimal transport-based clustering optimization module, achieving optimal delineation of cell populations while maintaining high biological relevance.

<img width="581" alt="image" src="https://github.com/user-attachments/assets/e8765bcb-048b-418e-a2be-e9e7f893f279" />

# Conclusion
By integrating dual-channel cut-informed soft graph representation learning, a ZINB-based feature autoencoder, and optimal transport-driven clustering optimization, scSGC effectively overcomes the challenges of traditional hard graph constructions. Extensive experiments across 8 datasets demonstrate that scSGC outperforms 11 state-of-the-art clustering models in clustering accuracy, cell type annotation, and computational efficiency. These results highlight its substantial potential to advance scRNA-seq data analysis and deepen our understanding of cellular heterogeneity.

In conclusion, we propose scSGC, an efficient and accurate framework for clustering single-cell RNA sequencing data. 
By integrating dual-channel soft graph representation learning with deep cut-informed techniques and incorporating ZINB-based feature autoencoder and optimal transport-driven clustering optimization, scSGC effectively addresses the critical challenges associated with traditional hard graph constructions, improving clustering accuracy while preserving biological relevance. 
Extensive experiments across eight datasets demonstrate that scSGC significantly outperforms eleven state-of-the-art clustering models in terms of clustering accuracy, cell type annotation, and computational efficiency, highlighting its significant advantages in single-cell bioinformatics and cellular heterogeneity analysis.

Looking ahead, we plan to enhance scSGC by integrating advanced large language models and extending its applicability to diverse multi-omics data types, such as spatial transcriptomics, which offer valuable contextual and spatial information for understanding complex biological systems.
Moreover, future work will focus on improving the model's scalability and enhancing the interpretability of clustering results to support more intricate biological research and analyses, thereby opening new pathways for understanding cellular systems and promoting personalized medicine.


# Run Example
```shell
python train_scSGC.py --dataname 'Maayan_Mouse_Pancreas_cell_1' --num_class 13 --epochs 200 --foldername 'logger_folder' --gpu 0 --learning_rate 5e-3 --weight_decay 5e-3 --balancer 0.5 --factor_ort 25 --factor_KL 1e-3 --factor_corvar 1 --factor_construct 0.23 --factor_zinb 20 --highly_genes 1500
```
Here, we give the hyperparameters used for the Maayan_Mouse_Pancreas_cell_1 dataset. The hyperparameters for the rest of the datasets are found in the file train_scSGC.py.

If you want to replicate our experimental results, please use the hyperparameters we provided.

Please contact us if you encounter problems during the replication process.


# Requirements
We implement scSGC in Python 3.7 based on PyTorch (version 1.12+cu113).

```shell
Keras --- 2.4.3
njumpy --- 1.19.5
pandas --- 1.3.5
Scanpy --- 1.8.2
torch --- 1.12.0
```

Please note that if using different versions, the results reported in our paper might not be able to repeat.

# The raw data
Setting data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix and Y is the true labels), n_clusters to the number of clusters.

In order to ensure the accuracy of the experimental results, we conducted more than 10 times runs on all the datasets and reported the mean and variance of these running results, reducing the result bias caused by randomness and variability, so as to obtain more reliable and stable results. Hyperparameter settings for all datasets can be found in the code.
The final output reports the clustering performance, here is an example on Maayan_Mouse_Pancreas_cell_1 data:

Final: ACC= 0.9124, NMI= 0.8578, ARI= 0.9163

The raw data used in this paper can be found:https://github.com/XPgogogo/scSGC/tree/master/datasets


<img width="594" alt="image" src="https://github.com/user-attachments/assets/93974574-fa69-474f-968f-f5f2d0b5ea08" />




# Please cite our paper if you use this code or or the dataset we provide in your own work:

```
XXX
```

# Contact
Ph.D student Ping XU

Computer Network Information Center, Chinese Academy of Sciences

University of Chinese Academy of Sciences

No.2 Dongshen South St

Beijing, P.R China, 100190

Personal Email: xuping0098@gmail.com

Official Email: xuping@cnic.cn
