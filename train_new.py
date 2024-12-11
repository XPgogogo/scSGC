# -*- encoding: utf-8 -*-

import os
import argparse
import random
from loguru import logger
import numpy as np
import pickle
import torch
from sklearn.cluster import KMeans
from torchmetrics.functional import pairwise_cosine_similarity

from model import AE_GAT, FULL, AE_NN, FULL_NN, ClusterAssignment
# from utils import pdf_norm, estimate_kappa, evaluation, visual, target_distribution, get_laplace_matrix
from utils import get_laplace_matrix,construct_graph,normalize_adj,dataset_show_details,zinb_loss, evaluation,target_distribution,sinkhorn,print_parameters
import torch.nn as nn
import warnings
import torch.nn.functional as Fcex
import scanpy as sc
from preprocess import *
import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
# from utils import *

from time import time
warnings.filterwarnings('ignore')


label_dataset_1 = ['Xiaoping_mouse_bladder_cell','Junyue_worm_neuron_cell',
                   'Grace_CITE_CBMC_counts_top2000','Sonya_HumanLiver_counts_top5000']

dataset_all = ['Maayan_Mouse_Pancreas_cell_1','Maayan_Mouse_Pancreas_cell_2',
               'Maayan_Human_Pancreas_cell_2','Maayan_Human_Pancreas_cell_1',
               'Meuro_human_Pancreas_cell','Xiaoping_mouse_bladder_cell',
               'Maayan_Human_Pancreas_cell_3','Junyue_worm_neuron_cell',
               'Grace_CITE_CBMC_counts_top2000','Sonya_HumanLiver_counts_top5000']
dataset_num_clusters = [13,13,14,14,9,16,14,10,15,11]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description')
    # parser.add_argument('--dataname', default='Xiaoping_mouse_bladder_cell', type=str)
    # parser.add_argument('--num_class', default=16, type=int, help='number of classes')
    parser.add_argument('--dataname', default='Maayan_Mouse_Pancreas_cell_1', type=str)
    parser.add_argument('--num_class', default=13, type=int, help='number of classes')
    parser.add_argument('--gpu', default=2, type=int)


    embedding_num = 16
    parser.add_argument('--dims_encoder', default=[256, embedding_num], type=list)
    parser.add_argument('--dims_decoder', default=[embedding_num, 256], type=list)

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lambdas', default=5, type=float)
    parser.add_argument('--balancer', default=0.5, type=float)

    parser.add_argument('--factor_ort', default=1, type=float)
    parser.add_argument('--factor_KL', default=0.5, type=float)
    parser.add_argument('--factor_corvar', default=0.05, type=float)
    parser.add_argument('--factor_zinb', default=0.1, type=float)
    parser.add_argument('--highly_genes', default = 1500, type=int)
    

    parser.add_argument('--pretrain_model_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_centers_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_pseudo_labels_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_model_load_path', default='pkl', type=str)
    parser.add_argument('--pretrain_centers_load_path', default='pkl', type=str)
    parser.add_argument('--pretrain_pseudo_labels_load_path', default='pkl', type=str)

    parser.add_argument('--foldername', default='MAIN_modified', type=str)
    parser.add_argument('--show_details', default=True, type=bool)
    

    args = parser.parse_args()
    args.pretrain_model_save_path = 'result/{}/{}_model.pkl'.format(args.foldername, args.dataname)
    args.pretrain_centers_save_path = 'result/{}/{}_centers.pkl'.format(args.foldername, args.dataname)
    args.pretrain_pseudo_labels_save_path = 'result/{}/{}_pseudo_labels.pkl'.format(args.foldername, args.dataname)
    args.pretrain_model_load_path = 'result/{}/{}_model.pkl'.format(args.foldername, args.dataname)
    args.pretrain_centers_load_path = 'result/{}/{}_centers.pkl'.format(args.foldername, args.dataname)
    args.pretrain_pseudo_labels_load_path = 'result/{}/{}_pseudo_labels.pkl'.format(args.foldername, args.dataname)
    
    if os.path.isdir('result/{}/'.format(args.foldername)) == False:
        os.makedirs('result/{}/'.format(args.foldername))
    if os.path.isdir('log/{}/'.format(args.foldername)) == False:
        os.makedirs('log/{}/'.format(args.foldername))

    if args.dataname == 'Maayan_Mouse_Pancreas_cell_1':
        # seed=3407,epoch=200,machine=17
        args.learning_rate = 1e-3
        args.weight_decay = 5e-3
        args.balancer = 0.5
        args.factor_ort =20
        args.factor_KL = 5e-3
        args.factor_corvar = 1
        args.factor_construct = 0.23
        args.factor_zinb = 20
        args.highly_genes = 1500

    if args.dataname == 'Maayan_Mouse_Pancreas_cell_2':
        # seed=296,epoch=100,machine=38
        args.learning_rate = 1e-3
        args.weight_decay = 1e-3
        args.balancer = 0.7
        args.factor_ort =25
        args.factor_KL = 1e-3
        args.factor_corvar = 1
        args.factor_construct = 0.23
        args.factor_zinb = 25
        args.highly_genes = 1500

    if args.dataname == 'Maayan_Human_Pancreas_cell_2':
        # seed=3407,epoch=50,machine=38
        args.learning_rate = 1e-3
        args.weight_decay = 5e-4
        args.balancer = 0.7
        args.factor_ort =10
        args.factor_KL = 1e-3
        args.factor_corvar = 1
        args.factor_construct = 0.23
        args.factor_zinb = 25
        args.highly_genes = 2000

    if args.dataname == 'Maayan_Human_Pancreas_cell_1':
        # seed=3407,epoch=150,machine=38
        args.learning_rate = 1e-3
        args.weight_decay = 5e-4
        args.balancer = 0.7
        args.factor_ort =30
        args.factor_KL = 1e-3
        args.factor_corvar = 1
        args.factor_construct = 0.23
        args.factor_zinb = 25
        args.highly_genes = 2000

    if args.dataname == 'Meuro_human_Pancreas_cell':
        # seed=233,epoch=50,machine=38
        args.learning_rate = 1e-3
        args.weight_decay = 5e-3
        args.balancer = 0.5
        args.factor_ort =10
        args.factor_KL = 1e-3
        args.factor_corvar = 1
        args.factor_construct = 0.23
        args.factor_zinb = 100
        args.highly_genes = 2000

    if args.dataname == 'Xiaoping_mouse_bladder_cell':
        # seed=3407,epoch=200
        args.learning_rate = 1e-3
        args.weight_decay = 5e-3
        args.balancer = 0.5
        args.factor_ort =10
        args.factor_KL = 1e-3
        args.factor_corvar = 1
        args.factor_construct = 0.23
        args.factor_zinb = 50
        args.highly_genes = 2000
    if args.dataname == 'Maayan_Human_Pancreas_cell_3':
        # seed=2021,epoch=200,machine=17
        args.learning_rate = 1e-3
        args.weight_decay = 5e-4
        args.balancer = 0.1
        args.factor_ort =30
        args.factor_KL = 1e-3
        args.factor_corvar = 1
        args.factor_construct = 0.23
        args.factor_zinb = 25
        args.highly_genes = 2000
    if args.dataname == 'Junyue_worm_neuron_cell':
        # seed=3407,epoch=200
        args.learning_rate = 5e-4
        args.weight_decay = 1e-3
        args.balancer = 0.7
        args.factor_ort = 0.95
        args.factor_KL = 0.48
        args.factor_corvar = 0.12
        args.factor_construct = 0.6
        args.factor_zinb = 0.1
        args.highly_genes = 1500
    if args.dataname == 'Grace_CITE_CBMC_counts_top2000':
        # seed=3407,epoch=200,machine=38
        args.learning_rate = 1e-3
        args.weight_decay = 5e-3
        args.balancer = 0.7
        args.factor_ort =10
        args.factor_KL = 1e-3
        args.factor_corvar = 1
        args.factor_construct = 0.23
        args.factor_zinb = 50
        args.highly_genes = 1500
    if args.dataname == 'Sonya_HumanLiver_counts_top5000':
        # seed=2486,epoch=200,machine=38
        args.learning_rate = 1e-3
        args.weight_decay = 5e-3
        args.balancer = 0.7
        args.factor_ort =15
        args.factor_KL = 1e-3
        args.factor_corvar = 0
        args.factor_construct = 0.23
        args.factor_zinb = 50
        args.highly_genes = 1500


    print_parameters(args.dataname,args.learning_rate,args.weight_decay,args.balancer,args.factor_corvar,args.factor_ort,args.factor_zinb,args.factor_KL)

    logger.add('log/{}/{}.log'.format(args.foldername, args.dataname), rotation="500 MB", level="INFO")
    logger.info(args)
    
    torch.cuda.set_device(args.gpu)
    
    datapath = os.path.join('/home/xuping/scRNA-seq_GraphClustering/scCDCG_BIB/datasets/', args.dataname) 
    embd_file_path = os.path.join('/home/xuping/scRNA-seq_GraphClustering/scCDCG_BIB/embd/', args.dataname+'.h5')
        
    if args.dataname == 'Meuro_human_Pancreas_cell':
        x, y = prepro(datapath+'.h5')
    else:
        data = h5py.File(datapath+'.h5','r')
        x = data['X'][:]
        y = data['Y'][:]

    if args.dataname == 'Meuro_human_Pancreas_cell':
        x =  np.round(x).astype(int)
    if args.dataname in label_dataset_1:
        y = y-1
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    ############
    # 使用 scanpy 挑选高表达基因组建新的特征矩阵X，原始矩阵为X_raw,计算sf。
    adata = sc.AnnData(x.numpy())
    adata.obs['Group'] = y
    # count_X = x
    adata = normalize_sc(adata, 
                      copy=True, 
                      highly_genes=args.highly_genes, 
                      size_factors=True, 
                      normalize_input=True, 
                      logtrans_input=True)
    
    X = adata.X.astype(np.float32)
    X_raw = adata.raw.X #原始数据，没有抽取高表达基因的二维数组
    sf = adata.obs.size_factors
    Y = np.array(adata.obs["Group"])

    # high_variable = np.array(adata.var.highly_variable.index, dtype=np.int) #高可变的500个基因对应的索引
    # count_X = count_X[:, high_variable] # 抽离出高表达的500个基因的全基因组表达数据


    #############
    # Model Input：高表达基因矩阵；sf(实验证明这个的效果更好)
    # X = torch.from_numpy(X)
    # X_ = torch.nn.functional.normalize(x, p=2, dim=1)   
    ############
    # # # # Model Input：未做任何处理的基因表达矩阵；sf

    # X = torch.from_numpy(x.cpu().numpy() )
    # X_ = torch.nn.functional.normalize(x, p=2, dim=1)   

    
    # # 构图
    # # heat p cos ncos
    # adj_1 = construct_graph(X_, Y, "cos", topk=10)
    # adj_norm_1 = normalize_adj(adj_1, self_loop=True, symmetry=True) #包含symmetric+self-loop

    # adj_2 = construct_graph(X_, Y, "ncos", topk=10)
    # adj_norm_2 = normalize_adj(adj_norm_1, self_loop=True, symmetry=True) #包含symmetric+self-loop

    # L_1 = get_laplace_matrix(adj_norm_1)
    # L_2 = get_laplace_matrix(adj_norm_2)
    # if args.show_details  is True:
    #     dataset_show_details(args.dataname,x_,y,adj_1)

    # # 相似度矩阵
    X = torch.from_numpy(X)
    X_ = torch.nn.functional.normalize(X, p=2, dim=1)  

    adj_1 = torch.mm(X_, X_.T)
    adj_2 = np.abs(pairwise_cosine_similarity(X_, X_))
    adj_2 = torch.mm(adj_2, adj_2.T)
    
    L_1 = get_laplace_matrix(adj_1)
    L_2 = get_laplace_matrix(adj_2)


    # for seed in [2021,2022,2050,3047,3041]:
    # for seed in [3407,3041,2021,2022,2050,3047]:
    for seed in [3407]:
    # for seed in [random.randint(0, 5000) for  i in range(100)]:
        logger.info('Seed {}'.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        pre_start_time = time()

        # ################################### PRE-TRAIN ########################################
        # Model = AE_GAT(dim_input=x.shape[1], dims_encoder=args.dims_encoder, dims_decoder=args.dims_decoder).cuda()
        Model = AE_NN(dim_input=X.shape[1], dims_encoder=args.dims_encoder, dims_decoder=args.dims_decoder).cuda()
        optimizer = torch.optim.Adam(Model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        acc_max = 0
        for epoch in range(1, args.epochs+1):

            # h, x_hat = Model.forward(
            # x_.cuda())
            h,x_hat,meanbatch, dispbatch, pibatch = Model.forward(X.cuda())
            z = torch.nn.functional.normalize(h, p=2, dim=0)
            # adj_pred = torch.mm(z, z.T)

            # loss_x = torch.nn.functional.mse_loss(x_hat, x.cuda()) # MSE重构损失
            loss_corvariates = -torch.mm(torch.mm(z.T, (args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),z).trace()/len(z.T)
            loss_ort =  torch.nn.functional.mse_loss(torch.mm(z.T,z).view(-1).cuda(),torch.eye(len(z.T)).view(-1).cuda())
            # loss_zinb = zinb_loss(x_,meanbatch, dispbatch, pibatch, sf)
            loss_zinb = zinb_loss(X.cuda(), meanbatch.cuda(), dispbatch.cuda(), pibatch.cuda(), sf, device='cuda')

            # loss = args.factor_construct * loss_x + args.factor_ort * loss_ort + args.factor_corvar * loss_corvariates
            # loss = args.factor_ort * loss_ort + args.factor_corvar * loss_corvariates #没有解码器的重构损失+ZINBloss
            loss = args.factor_ort * loss_ort + args.factor_corvar * loss_corvariates + args.factor_zinb * loss_zinb


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                kmeans = KMeans(n_clusters=args.num_class, random_state=2021, n_init=20).fit(z.cpu().numpy())
                acc, nmi, ari, f1_macro = evaluation(y, kmeans.labels_)
                centers = torch.tensor(kmeans.cluster_centers_)

                logger.info('Epoch {}/{} Pre-Train ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(epoch, args.epochs, acc, nmi, ari, f1_macro))
                logger.info('Epoch {}/{} | loss_corvariate: {:.6f} | loss_ort: {:.6f} | loss_zinb: {:.6f} | loss_total: {:.6f}'.format(epoch, args.epochs, loss_corvariates.cpu().item(), loss_zinb.cpu().item(), loss_ort.cpu().item(),  loss.cpu().item()))

                if acc > acc_max:
                    acc_max = acc
                    torch.save(Model.state_dict(), args.pretrain_model_save_path)
                    with open(args.pretrain_centers_save_path,'wb') as save1:
                        pickle.dump(centers, save1, protocol=pickle.HIGHEST_PROTOCOL)
                    pseudo_labels = torch.LongTensor(kmeans.labels_)
                    with open(args.pretrain_pseudo_labels_save_path,'wb') as save2:
                        pickle.dump(pseudo_labels, save2, protocol=pickle.HIGHEST_PROTOCOL)
        pro_time = time() - pre_start_time
        train_start_time = time()

    
        ####################################### TRAIN ########################################
        # Model = FULL(dim_input=x.shape[1], dims_encoder=args.dims_encoder, dims_decoder=args.dims_decoder, num_class=args.num_class, \
        #             pretrain_model_load_path=args.pretrain_model_load_path).cuda()
        Model = FULL_NN(dim_input=X.shape[1], dims_encoder=args.dims_encoder, dims_decoder=args.dims_decoder, num_class=args.num_class, \
                    pretrain_model_load_path=args.pretrain_model_load_path).cuda()

        # 假设模型中有两个参数 W 和 b
        # 假设model是你的模型实例
        # for name, param in Model.named_parameters():
        #     print(f"Parameter name: {name}, Shape: {param.shape}")

        optimizer = torch.optim.Adam(Model.parameters(), lr=args.learning_rate)
        with open(args.pretrain_centers_load_path,'rb') as load1:
            centers = pickle.load(load1).cuda()
        with open(args.pretrain_pseudo_labels_load_path,'rb') as load2:
            pseudo_labels = pickle.load(load2).cuda()

        acc_max, nmi_max, ari_max, f1_macro_max = 0, 0, 0, 0
        for epoch in range(1, args.epochs+1):
            z , x_hat, meanbatch, dispbatch, pibatch = Model.forward(X.cuda())
            z = torch.nn.functional.normalize(z, p=2, dim=0)
            centers = centers.detach()
            # adj_pred = torch.mm(z, z.T)
            # loss_x = torch.nn.functional.mse_loss(x_hat, X_.cuda()) # MSE重构损失
            loss_corvariates = -torch.mm(torch.mm(z.T, ( args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),z).trace()/len(z.T)
            loss_ort = torch.nn.functional.mse_loss(torch.mm(z.T,z).view(-1).cuda(),torch.eye(len(z.T)).view(-1).cuda())
            # loss_zinb = zinb_loss(x_,meanbatch, dispbatch, pibatch, sf)
            loss_zinb = zinb_loss(X.cuda(), meanbatch.cuda(), dispbatch.cuda(), pibatch.cuda(), sf, device='cuda')
            # loss_adj_graph = torch.nn.functional.mse_loss(adj_pred.view(-1), adj_norm.cuda().view(-1))
       
            #### DEC 
            class_assign_model = ClusterAssignment(args.num_class, len(z.T), 1, centers).cuda()
            temp_class = class_assign_model(z.cuda())
            ### target function
            # if epoch == 1:
            #     p_distribution = target_distribution(temp_class).detach()
            # if epoch // 10 == 0:
            #     p_distribution = target_distribution(temp_class).detach()

            #### sinkhole
            if epoch == 1:
                p_distribution = torch.tensor(sinkhorn ( temp_class.cpu().detach().numpy(), args.lambdas, torch.ones(x.shape[0]).numpy(), torch.tensor([torch.sum(pseudo_labels==i) for i in range(args.num_class)]).numpy())).float().cuda().detach()
                p_distribution = p_distribution.detach()
                q_max, q_max_index = torch.max(p_distribution, dim=1)
            elif epoch // 10 == 0:
                p_distribution = torch.tensor(sinkhorn ( temp_class.cpu().detach().numpy(), args.lambdas, torch.ones(x.shape[0]).numpy(), torch.tensor([torch.sum(pseudo_labels==i) for i in range(args.num_class)]).numpy())).float().cuda().detach()
                p_distribution = p_distribution.detach()
                q_max, q_max_index = torch.max(p_distribution, dim=1)

            KL_loss_function = nn.KLDivLoss(reduction='sum') 
            loss_KL = KL_loss_function(temp_class.cuda(), p_distribution.cuda()) / temp_class.shape[0]
            
            # KL_loss_function = nn.KLDivLoss(reduction='mean')
            # loss_KL = KL_loss_function(temp_class.cuda(), p_distribution.cuda())

            # loss = args.factor_construct * loss_x + args.factor_ort * loss_ort + args.factor_corvar * loss_corvariates + args.factor_KL * loss_KL
            # loss = args.factor_corvar * loss_corvariates + args.factor_ort * loss_ort + args.factor_KL * loss_KL +
            loss = args.factor_corvar * loss_corvariates + args.factor_ort * loss_ort + args.factor_KL * loss_KL + args.factor_zinb * loss_zinb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                kmeans = KMeans(n_clusters=args.num_class, random_state=2021, n_init=20).fit(z.cpu().numpy())
                # kmeans = KMeans(n_clusters=args.num_class, random_state=2021, n_init=20).fit(np.nan_to_num(z.cpu().numpy()))
                acc, nmi, ari, f1_macro = evaluation(y, kmeans.labels_)
                if acc_max < acc:
                    acc_max, nmi_max, ari_max, f1_macro_max = acc, nmi, ari, f1_macro
                    with h5py.File(embd_file_path, 'w') as file:
                        file.create_dataset('X', data=z.cpu().numpy())
                        file.create_dataset('Y', data = kmeans.labels_)
                pseudo_labels = torch.LongTensor(kmeans.labels_)
                centers = torch.tensor(kmeans.cluster_centers_)
                
                #### logger
                logger.info('Epoch {}/{} | loss_corvariate: {:.6f} | loss_ort: {:.6f} | loss_zinb: {:.6f} | loss_KL: {:.6f} | loss_total: {:.6f}'.format(epoch, args.epochs, loss_corvariates.cpu().item(), loss_ort.cpu().item(), loss_zinb.cpu().item(), loss_KL.cpu().item(),  loss.cpu().item()))
                logger.info('Epoch {}/{} ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(epoch, args.epochs, acc, nmi, ari, f1_macro))
        logger.info('MAX ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(acc_max, nmi_max, ari_max, f1_macro_max))
        

        train_time = time() - train_start_time
        all_time = time() - pre_start_time
        logger.info('dataset_name:{},pre_time:{:.7f},train_time:{:.7f},all_time:{:.7f}'.format(args.dataname,pro_time,train_time,all_time))
        logger.info('Average_time,pre_time:{:.7f},train_time:{:.7f},all_time:{:.7f}'.format(pro_time/args.epochs,train_time/args.epochs,all_time/args.epochs))
        logger.info('seed:{}, dataset_name:{}, learning_rate:{}, weight_decay:{}, balancer:{}, factor_ort:{}, factor_KL:{}, factor_corvar:{}, factor_zinb:{},highly_genes:{}'.format(seed, args.dataname, args.learning_rate, args.weight_decay, args.balancer, args.factor_ort, args.factor_KL, args.factor_corvar, args.factor_zinb, args.highly_genes))



