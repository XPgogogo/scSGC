# scSGC
scSGC, a clustering model based on dual-channel cut-informed soft graph for scRNA-seq data. See details in our paper: "Soft Graph Clustering for single-cell Sequencing Data" published in XXXXXX（CCF-X）.
（Accepted as a XXX paper for the research track at XXXXXX）

(arXiv: https:    )

（DOI：      ）


# Overview
Clustering analysis plays a key role in single-cell RNA sequencing (scRNA-seq) data analysis for elucidating cellular heterogeneity and diversity. Recent graph-based scRNA-seq clustering methods, particularly graph neural networks (GNNs), have significantly improved in tackling the challenges of high-dimension, high-sparsity, and frequent dropout events that lead to ambiguous cell population boundaries. However, the GNN-based method is intended for general graph encoding and faces challenges when applied to scRNA-seq data due to the following reasons: (i) GNN-based methods typically construct hard graphs from similarity matrices by applying a threshold that overly simplifies intercellular relationships into binary edges (0 or 1), which restricts the capture of continuous similarity features among cells and leads to significant information loss. (ii) Hard graphs derived from scRNA-seq data, which often exhibit significant inter-cluster connections, present challenges for GNN-based methods that rely heavily on underlying graph structures and typically generate similar representations for neighboring nodes, thereby leading to erroneous message propagation and biased clustering outcomes. 
To address these issues, scSGC introduces a Soft Graph Clustering framework composed of three key modules:(i) A ZINB-based feature autoencoder to effectively model scRNA-seq data distributions and mitigate challenges of high sparsity and dropout events;(ii) a dual-channel cut-informed soft graph embedding module, which captures continuous intercellular similarities while preserving global and local graph structures;(iii) an optimal transport-based clustering optimization module, ensuring accurate clustering with high biological relevance. 
Extensive experiments across 8 datasets demonstrate that scSGC significantly outperforms 11 state-of-the-art clustering models in terms of clustering accuracy, cell type annotation, and computational efficiency, highlighting its substantial potential to enhance scRNA-seq data analysis and advance understanding of cellular heterogeneity.
In summary, scSGC represents a major enhancement over existing methods, addressing critical challenges in scRNA-seq clustering while expanding on our earlier work.

<img width="1203" alt="image" src="https://github.com/user-attachments/assets/90f8204e-ce23-4491-b57f-32873e417805">


Fig.1(a) depicts a hard graph GNN-based framework for scRNA-seq clustering, while Fig.1(b) illustrates the framework of our proposed method, scSGC.
In contrast, scSGC offers two key advantages: (i) It tightly integrates two key modules within the graph-based scRNA-seq clustering framework, i.e., the feature autoencoder and the graph autoencoder, allowing both modules to optimize the final embedding collaboratively; (ii) By employing a soft graph construction strategy, it eliminates reliance on hard graph structures, enabling more effective capture of intracellular structural information and fully utilizing continuous similarities between cells.
Specifically, our proposed method, scSGC, comprises three key modules: (i) ZINB-based feature autoencoder, which employs a ZINB model to characterize scRNA-seq data specifically to address high sparsity and dropout rates in gene expression data; (ii) Cut-informed soft graph modeling (see Fig.1(c) for its architecture), leverages dual-channel cut-informed soft graph construction to generate consistent and optimized cellular representations, facilitating smoother capture of intercellular continuous structural information; (iii) Optimization via optimal transport, utilizing optimal transport theory, achieving optimal partitioning of cell populations at minimal transport cost and ensuring stable clustering results within complex data structures. 

In scSGC, we first model the raw scRNA-seq data using a ZINB autoencoder to generate robust cellular representations. Then, two soft graphs are constructed using the input data, and their corresponding laplacian matrices are computed. These matrices undergo a minimum jointly normalized cut through a graph-cut strategy to optimize the representation of cell-cell relationships. Finally, an optimal transport-based self-supervised learning approach is employed to refine the clustering, ensuring accurate partitioning of cell populations in high-dimensional and high-sparse data. 


# Run Example
```shell
python train_scSGC.py --dataname 'Maayan_Human_Pancreas_cell_1' --num_class 14 --epochs 200 --foldername 'logger_folder' --gpu 0 --learning_rate 1e-3 --weight_decay 5e-4 --balancer 0.7 --factor_ort 30 --factor_KL 1e-3 --factor_corvar 1 --factor_construct 0.23 --factor_zinb 25 --highly_genes 2000
```
Here, we give the hyperparameters used for the Maayan_Human_Pancreas_cell_1 dataset. The hyperparameters for the rest of the datasets are found in the file train_scSGC.py.

If you want to replicate our experimental results, please use the hyperparameters we provided.

Please contact us if you encounter problems during the replication process.

# Requirements
We implement scSGC in Python 3.7 based on PyTorch (version 1.12+cu113).

```shell
Keras --- 2.4.3
numpy --- 1.19.5
pandas --- 1.3.5
Scanpy --- 1.8.2
torch --- 1.12.0
```

Please note that if using different versions, the results reported in our paper might not be able to repeat.

# The raw data
Setting data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix and Y is the true labels), n_clusters to the number of clusters.

In order to ensure the accuracy of the experimental results, we conducted more than 10 times runs on all the datasets and reported the mean and variance of these running results, reducing the result bias caused by randomness and variability, so as to obtain more reliable and stable results. Hyperparameter settings for all datasets can be found in the code.
The final output reports the clustering performance, here is an example on Maayan_Human_Pancreas_cell_1 scRNA-seq data:

Final: ACC= 0.9625, NMI= 0.9142, ARI= 0.9489

The raw data used in this paper can be found:https://github.com/XPgogogo/scSGC/tree/master/datasets

![Uploading image.png…]()


# Please cite our paper if you use this code or or the dataset we provide in your own work:

```
@article{xu2024sccdcg,
  title={Soft Graph Clustering for single-cell Sequencing Data},
  author={},
  journal={},
  year={}
}
```

# Contact
Ph.D student Ping XU

Computer Network Information Center, Chinese Academy of Sciences

University of Chinese Academy of Sciences

No.2 Dongshen South St

Beijing, P.R China, 100190

Personal Email: xuping0098@gmail.com

Official Email: xuping@cnic.cn
