# -*- encoding: utf-8 -*-

# from layer import GAT_Layer
import torch
from utils import pdf_norm
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional
import torch.nn.functional as F


######################################## GAT Layer ########################################

# dim_in: dim_input , 256 , embedding_num
# dim_out: embedding_num , 256 , dim_imput
class GAT_Layer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, negative_slope=0.2):
        super(GAT_Layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w = torch.nn.Parameter(torch.FloatTensor(self.dim_in, self.dim_out))
        self.a_target = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        self.a_neighbor = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        torch.nn.init.xavier_normal_(self.w, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_target, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_neighbor, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, x, adj):
        # x (num_nodes, dim_in)
        x_ = torch.mm(x, self.w) # (num_nodes, dim_out)
        scores_target = torch.mm(x_, self.a_target) # (num_nodes, )
        scores_neighbor = torch.mm(x_, self.a_neighbor) # (num_nodes, )
        scores = scores_target + torch.transpose(scores_neighbor, 0, 1) # (num_nodes, num_nodes)
        
        scores = torch.mul(adj, scores)
        scores = self.leakyrelu(scores)
        scores = torch.where(adj>0, scores, -9e15*torch.ones_like(scores)) # score of non-negihbor is -9e15
        coefficients = torch.nn.functional.softmax(scores, dim=1)
        x_ = torch.nn.functional.elu(torch.mm(coefficients, x_))
        return x_
    
    
    

######################################## GAT Auto_Encoder ########################################


# dims_en: dim_input , 256 , embedding_num
# dims_de: embedding_num , 256 , dim_imput
# num_layer: 2

class AE_GAT(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(AE_GAT, self).__init__()
        self.dims_en = [dim_input] + dims_encoder
        self.dims_de = dims_decoder + [dim_input]

        self.num_layer = len(self.dims_en)-1
        # print("self_numlayer", self.num_layer)
        self.Encoder = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        for index in range(self.num_layer):
            # print("layer:", index, self.dims_en[index], self.dims_en[index+1] )
            # print("layer:", index, self.dims_de[index], self.dims_de[index+1] )
            self.Encoder.append(GAT_Layer(self.dims_en[index], self.dims_en[index+1]))
            self.Decoder.append(GAT_Layer(self.dims_de[index], self.dims_de[index+1]))

    def forward(self, x, adj):
        # print("x,adj:", x.shape, adj.shape)
        for index in range(self.num_layer):
            x = self.Encoder[index].forward(x, adj)
        h = x
        for index in range(self.num_layer):
            x = self.Decoder[index].forward(x, adj)
        x_hat = x      
        return h, x_hat
    

################### ZINB from scDSC #######################
# class ZINBLoss(nn.Module):
#     def __init__(self):
#         super(ZINBLoss, self).__init__()
#     def forward(self, X, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
#         eps = 1e-10
#         scale_factor = scale_factor[:, None]
#         mean = mean * scale_factor
#         t1 = torch.lgamma(disp+eps) + torch.lgamma(X+1.0) - torch.lgamma(X+disp+eps)
#         # print('t1')
#         t2 = (disp+X) * torch.log(1.0 + (mean/(disp+eps))) + (X * (torch.log(disp+eps) - torch.log(mean+eps)))
#         nb_final = t1 + t2

#         nb_case = nb_final - torch.log(1.0-pi+eps)
#         zero_nb = torch.pow(disp/(disp+mean+eps), disp)
#         zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
#         result = torch.where(torch.le(X, 1e-8), zero_case, nb_case)
        
#         if ridge_lambda > 0:
#             ridge = ridge_lambda*torch.square(pi)
#             result += ridge
#         result = torch.mean(result)
#         return result
    
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, X):
        return torch.clamp(torch.exp(X), min=1e-5, max=1e6)
    

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, X):
        return torch.clamp(F.softplus(X), min=1e-4, max=1e4)


######################################## NN Auto_Encoder ########################################

# dims_en [1870, 256, 16] [16, 256, 1870]
# self_numlayer 2
# layer: 0 1870 256
# layer: 0 16 256
# layer: 1 256 16
# layer: 1 256 1870

# dims_en: dim_input , 256 , embedding_num
# dims_de: embedding_num , 256 , dim_imput
# num_layer: 2

class AE_NN(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(AE_NN, self).__init__()
        self.dims_en = [dim_input] + dims_encoder
        self.dims_de = dims_decoder + [dim_input]

        self.num_layer = len(self.dims_en)-1
        # print("self_numlayer", self.num_layer)
        
        self.Encoder = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        for index in range(self.num_layer):
            self.Encoder.append(torch.nn.Linear(self.dims_en[index], self.dims_en[index+1]))
            self.Decoder.append(torch.nn.Linear(self.dims_de[index], self.dims_de[index+1]))

        self._dec_mean = nn.Sequential(nn.Linear(dims_decoder[-1], dim_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(dims_decoder[-1], dim_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(dims_decoder[-1], dim_input), nn.Sigmoid())
        

    def forward(self, x):
        # 编码
        for index in range(self.num_layer):
            x = self.Encoder[index](x)
            x = self.leakyrelu(x)
        embedding = x #中间的隐藏层
        
        # 解码
        for index in range(self.num_layer-1):
            x = self.Decoder[index](x)
            x = self.leakyrelu(x)
        input_to_zinb_embd = x

        _mean = self._dec_mean(input_to_zinb_embd)
        _disp = self._dec_disp(input_to_zinb_embd)
        _pi = self._dec_pi(input_to_zinb_embd)

        x_hat = self.Decoder[range(self.num_layer)[-1]](x)   
  
        # return embedding, x_hat
        return embedding, x_hat, _mean, _disp, _pi



####################################### FULL ########################################
class FULL(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(FULL, self).__init__()
        self.dims_encoder = dims_encoder
        self.num_class = num_class

        self.AE = AE_GAT(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(torch.load(pretrain_model_load_path, map_location='cpu')) # initialization with pretrain auto_encoder
    
    def forward(self, x, adj):
        h, x_hat = self.AE.forward(x, adj)
        self.z = torch.nn.functional.normalize(h, p=2, dim=1)
        return self.z, x_hat

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T)) # (num_nodes, num_class)
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = torch.nn.functional.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        return p


####################################### FULL NN ########################################
class FULL_NN(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(FULL_NN, self).__init__()
        self.dims_encoder = dims_encoder
        self.num_class = num_class

        self.AE = AE_NN(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(torch.load(pretrain_model_load_path, map_location='cpu')) # initialization with pretrain auto_encoder
  
    def forward(self, x):
        h, x_hat, _mean, _disp, _pi = self.AE.forward(x)
        self.z = torch.nn.functional.normalize(h, p=2, dim=1)
        # print('x_hat.shape:', x_hat.shape)
        # Dual Self-supervised Module
 
        # return self.z, x_hat, _mean, _disp, _pi
        return self.z, x_hat, _mean, _disp, _pi
 

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T)) # (num_nodes, num_class)
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = torch.nn.functional.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        return p


####################################### ClusterAssignment ########################################
class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
