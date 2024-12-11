# -*- encoding: utf-8 -*-

import torch
import math
import numpy as np
import scipy.sparse as sp
from scipy.special import iv
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import recall_score, precision_score

################################################################################
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

######################################## Evaluation ########################################
def best_map(y_true, y_pred):
    """
    https://github.com/jundongl/scikit-feature/blob/master/skfeature/utility/unsupervised_evaluation.py
    Permute labels of y_pred to match y_true as much as possible
    """
    if len(y_true) != len(y_pred):
        print("y_true.shape must == y_pred.shape")
        exit(0)

    label_set = np.unique(y_true)
    num_class = len(label_set)

    G = np.zeros((num_class, num_class))
    for i in range(0, num_class):
        for j in range(0, num_class):
            s = y_true == label_set[i]
            t = y_pred == label_set[j]
            G[i, j] = np.count_nonzero(s & t)

    A = linear_assignment(-G)
    new_y_pred = np.zeros(y_pred.shape)
    for i in range(0, num_class):
        new_y_pred[y_pred == label_set[A[1][i]]] = label_set[A[0][i]]
    return new_y_pred.astype(int), label_set[A[1]], label_set[A[0]]

def evaluation(y_true, y_pred):
    y_pred_, label_original, label_truth = best_map(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_)
    f1_macro = f1_score(y_true, y_pred_, average='macro')
    # f1_micro = f1_score(y_true, best_map(y_true, y_pred), average='micro')
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # print('origi label', label_original)
    # print('truth label', label_truth)
    # print('recall', recall_score(y_true, y_pred_, average=None))
    # print('precision', precision_score(y_true, y_pred_, average=None))
    return acc, nmi, ari, f1_macro

######################################## vMF ########################################
def pdf_norm(dim, kappas):
    numerator = torch.pow(kappas, dim/2 -1)
    denominator = torch.pow(torch.mul(torch.pow(torch.ones_like(kappas)*2*math.pi, dim/2), iv(dim/2 -1, kappas)), -1)
    return torch.mul(numerator, denominator)

def A_d(dim, kappas):
    numerator = iv(dim/2, kappas)
    denominator = torch.pow(iv(dim/2 -1, kappas), -1)
    return torch.mul(numerator, denominator)

def estimate_kappa(dim, kappas):
    r = A_d(dim, kappas)
    numerator = dim*r - torch.pow(r, 3)
    denominator = torch.pow(1 - torch.pow(r, 2), -1)
    return torch.mul(numerator, denominator)

######################################## Visual ########################################
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.family'] = 'Times New Roman'
def visual(num_class, h, y, c, pred_q, save_path_truth, save_path_pred_q):
    h = np.vstack((h, c))
    # h_ = TSNE(n_components=2, init='pca', random_state=0, early_exaggeration=30).fit_transform(h)
    pca = PCA(n_components=2)
    h_ = pca.fit_transform(h)
    h = h_[:-c.shape[0]]
    c = h_[-c.shape[0]:]

    # # h_ = TSNE(n_components=2, init='pca', random_state=0, early_exaggeration=30).fit_transform(h)
    # pca = PCA(n_components=2)
    # h = pca.fit_transform(h)
    # c = pca.fit_transform(c)

    fig, ax = plt.subplots()
    # plt.xlim(-1.25, 1.25)
    # plt.ylim(-1.25, 1.25)
    for index, color in zip(range(num_class), ['tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:purple', 'yellow', 'navy', 'black', 'tan', 'cyan']):
        mask = (y[:]==index)
        axis_0 = h[:, 0][mask]
        axis_1 = h[:, 1][mask]
        ax.scatter(axis_0, axis_1, c=color, label='cluster '+str(index), s=10, alpha=1, edgecolors='none')
    # ax.grid(True)
    plt.axis('off')
    # ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig(save_path_truth, bbox_inches='tight')
    
    # fig, ax = plt.subplots()
    # # plt.xlim(-1.25, 1.25)
    # # plt.ylim(-1.25, 1.25)
    # for index, color in zip(range(num_class), ['tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:purple', 'yellow', 'navy', 'black', 'tan', 'cyan']):
    #     mask = (pred_p[:]==index)
    #     axis_0 = h[:, 0][mask]
    #     axis_1 = h[:, 1][mask]
    #     ax.scatter(axis_0, axis_1, c=color, label='cluster '+str(index), s=10, alpha=1, edgecolors='none')
    #     ax.scatter(c[index, 0], c[index, 1], c=color, label='center '+str(index), s=100, alpha=1, edgecolors='black')
    # # ax.grid(True)
    # ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    # plt.savefig(save_path_pred_p, bbox_inches='tight')

    fig, ax = plt.subplots()
    # plt.xlim(-1.25, 1.25)
    # plt.ylim(-1.25, 1.25)
    for index, color in zip(range(num_class), ['tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:purple', 'yellow', 'navy', 'black', 'tan', 'cyan']):
        mask = (pred_q[:]==index)
        axis_0 = h[:, 0][mask]
        axis_1 = h[:, 1][mask]
        ax.scatter(axis_0, axis_1, c=color, label='cluster '+str(index), s=10, alpha=1, edgecolors='none')
        ax.scatter(c[index, 0], c[index, 1], c=color, label='center '+str(index), s=100, alpha=1, edgecolors='black')
    # ax.grid(True)
    plt.axis('off')
    # ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig(save_path_pred_q, bbox_inches='tight')



    ####################################### SciPy ########################################
def obj_func(target, pred):
    target = target.reshape(pred.shape[0], pred.shape[1])
    loss = -np.mean(target * np.log(pred))
    return loss

def grad_func(target, pred):
    gradient = -np.log(pred)
    return np.ravel(gradient)

def cons_row(i, shape0, shape1):  
    return {'type':'eq', 'fun': lambda x: np.sum(x.reshape(shape0, shape1), axis=1)[i] - 1}  
def cons_col(j, shape0, shape1):
    return {'type':'eq', 'fun': lambda x: np.sum(x.reshape(shape0, shape1), axis=0)[j] - shape0/shape1} 
def cons_positive(k):
    return {'type':'ineq', 'fun': lambda x: x[k]}
def cons_orthogonal(j1, j2, shape0, shape1):
    return {'type':'eq', 'fun': lambda x: np.dot(x.reshape(shape0, shape1).T, x.reshape(shape0, shape1))[j1][j2]}

def re_assignment(pred):
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    cons_1 = list(map(cons_row, list(range(num_node)), [num_node for i in range(num_node)], [num_class for i in range(num_node)]))
    cons_2 = list(map(cons_col, list(range(num_class)), [num_node for i in range(num_class)], [num_class for i in range(num_class)]))
    cons_3 = list(map(cons_positive, list(range(num_node*num_class))))
    cons_4 = list(map(cons_orthogonal, np.nonzero(np.eye(num_class)-1)[0].tolist(), np.nonzero(np.eye(num_class)-1)[1].tolist(), [num_node for i in range(num_class*(num_class-1))], [num_class for i in range(num_class*(num_class-1))]))
    cons = cons_1 + cons_2 + cons_3

    init_target = np.ravel(np.ones_like(pred)/num_class)

    res = minimize(fun=obj_func, x0=init_target, args=pred, jac=grad_func, constraints=cons)
    return res.success, res.x.reshape(num_node, num_class)

####################################### Greenhorn ########################################
def dist_pho(a, b):
    return b - a + a * np.log(a/b)

def greenkhorn(pred):
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    p = np.power(pred, 1).T

    row = np.ones(num_node)
    col = np.ones(num_class)*(num_node/num_class)

    x = np.ones_like(row)
    y = np.ones_like(col)

    for index in range(1000):
        max_i = np.argmax(dist_pho(row, np.sum(p, axis=1)))
        max_j = np.argmax(dist_pho(col, np.sum(p, axis=0)))
        
        print(dist_pho(row[max_i], torch.sum(q, dim=1)[max_i]), dist_pho(col[max_j], torch.sum(q, dim=0)[max_j]))
        if dist_pho(row[max_i], torch.sum(q, dim=1)[max_i]) > dist_pho(col[max_j], torch.sum(q, dim=0)[max_j]) :
            x[max_i] = x[max_i] + row[max_i] / torch.sum(q, dim=1)[max_i]
        else:
            y[max_j] = y[max_j] + col[max_j] / torch.sum(q, dim=0)[max_j]
        q = torch.mm(torch.mul(p, torch.exp(x).unsqueeze(1)), torch.diag(torch.exp(y)))
    print(torch.sum(q, dim=1), torch.sum(q, dim=0))
    return q


##### DEC target distribution ########
def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


##### laplace matrix
def get_laplace_matrix(tensor_matrix):
    A = np.array(tensor_matrix)
    D = A.sum(axis=1)
    L_matrix = np.diag(D**(-0.5)).dot(A.dot(np.diag(D**(-0.5))))
    L_matrix = torch.tensor(L_matrix,dtype=torch.float)
    # print("L_matrix",torch.isnan(L_matrix))
    # labels_count = L_matrix.unique(return_counts=True)
    # print("label_count", torch.isnan(L_matrix).int().sum())
    return torch.nan_to_num(L_matrix)






###############################

def print_parameters(dataname,learning_rate,weight_decay,balancer,factor_corvar,factor_ort,factor_zinb,factor_KL):
    print("++++++++++++++++++++++++++++++")
    print("---details of parameters---")
    print("++++++++++++++++++++++++++++++")
    print(f"dataname:{dataname}")
    print(f"learning_rate:{learning_rate}")
    print(f"weight_decay:{weight_decay}")
    print(f"balancer:{balancer}")
    print(f"factor_corvar:{factor_corvar}")
    print(f"factor_ort:{factor_ort}")
    print(f"factor_zinb:{factor_zinb}")
    print(f"factor_KL:{factor_KL}")
    print("++++++++++++++++++++++++++++++")


# 20240722添加
def dataset_show_details(dataset_name,feat,label,adj):
    print("++++++++++++++++++++++++++++++")
    print("---details of graph dataset---")
    print("++++++++++++++++++++++++++++++")
    print("dataset name:   ", dataset_name)
    print("feature shape:  ", feat.shape)
    print("label shape:    ", label.shape)
    print("adj shape:      ", adj.shape)
    print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
    print("category num:          ", max(label)-min(label)+1)
    print("category distribution: ")
    for i in range(max(label)+1):
        print("label", i, end=":")
        print(len(label[np.where(label == i)]))
    print("++++++++++++++++++++++++++++++")


    # return feat, label, adj, sf

# 构图 
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
# def construct_graph(features, label, method, topk=200):
# 构图 
def construct_graph(features, label, method, topk=10):
    num = len(label)
    dist = None
    # Several methods of calculating the similarity relationship between samples i and j (similarity matrix Sij)
    if method == 'heat':
        dist = -0.5 * pairwise_distances(features, metric='manhattan') ** 2
        dist = np.exp(dist)

    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)

    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    adj = np.zeros_like(dist)
    counter = 0
    for i, v in enumerate(inds):
        for vv in v:
            if vv != i:
                adj[i, vv] = 1
                if label[vv] != label[i]:
                    counter += 1
    # Save adjacency matrix as 'adj.npy'
    print('method: {}, error rate: {}'.format(method, counter / (num * topk)))
    return adj

def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj

def diffusion_adj(adj, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    adj_tmp = adj + np.eye(adj.shape[0])

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    sqrt_d_inv = np.sqrt(d_inv)

    # calculate norm adj
    norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    if mode == "ppr":
        diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))

    return diff_adj


def remove_edge(A, similarity, remove_rate=0.1):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges based on cosine similarity of embedding
    n_node = A.shape[0]
    for i in range(n_node):
        A[i, torch.argsort(similarity[i].cpu())[:int(round(remove_rate * n_node))]] = 0

    # normalize adj
    Am = normalize_adj(A, self_loop=True, symmetry=False)
    return Am

def gaussian_noised_feature(X, device='cuda'):
    """
    add gaussian noise to the attribute matrix X
    Args:
        X: the attribute matrix
    Returns: the noised attribute matrix X_tilde
    """
    X = X.to(device)
    Noise = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(device)
    X_gaussion = X * Noise
    return X_gaussion


####################################
# ZINB Loss
# def zinb_loss(X, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0, device='cuda'):
#     eps = 1e-10
#     eps = 1e-10
#     scale_factor = scale_factor[:, None]
#     mean = mean * scale_factor

#     t1 = torch.lgamma(disp+eps) + torch.lgamma(X+1.0) - torch.lgamma(X+disp+eps)
#     # print('t1')
#     t2 = (disp+X) * torch.log(1.0 + (mean/(disp+eps))) + (X * (torch.log(disp+eps) - torch.log(mean+eps)))
#     nb_final = t1 + t2

#     nb_case = nb_final - torch.log(1.0-pi+eps)
#     zero_nb = torch.pow(disp/(disp+mean+eps), disp)
#     zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
#     result = torch.where(torch.le(X, 1e-8), zero_case, nb_case)
        
#     if ridge_lambda > 0:
#         ridge = ridge_lambda*torch.square(pi)
#         result += ridge
#     result = torch.mean(result)
#     return result

def zinb_loss(X, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0, device='cuda'):
    eps = 1e-10
    
    # Convert scale_factor to a torch tensor if it is not already
    if not isinstance(scale_factor, torch.Tensor):
        scale_factor = torch.tensor(scale_factor, dtype=torch.float32).to(device)
    else:
        scale_factor = scale_factor.to(device)
    
    scale_factor = scale_factor[:, None]

    # Ensure all tensors are on the same device
    X = X.to(device)
    mean = mean.to(device) * scale_factor
    disp = disp.to(device)
    pi = pi.to(device)

    t1 = torch.lgamma(disp + eps) + torch.lgamma(X + 1.0) - torch.lgamma(X + disp + eps)
    t2 = (disp + X) * torch.log(1.0 + (mean / (disp + eps))) + (X * (torch.log(disp + eps) - torch.log(mean + eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0 - pi + eps)
    zero_nb = torch.pow(disp / (disp + mean + eps), disp)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.le(X, 1e-8), zero_case, nb_case)
    
    if ridge_lambda > 0:
        ridge = ridge_lambda * torch.square(pi)
        result += ridge
        
    result = torch.mean(result)
    return result


# def zinb_loss(X, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
#     eps = 1e-10
#     scale_factor = scale_factor[:, None]
#     mean = mean.detach().cpu().numpy() * scale_factor
#     mean = torch.from_numpy(mean)
#     # mean = mean * scale_factor
#     t1 = torch.lgamma(disp+eps) + torch.lgamma(X+1.0) - torch.lgamma(X+disp+eps)
#     t2 = (disp + X) * torch.log(1.0 + (mean / (disp + eps))) + (X * (torch.log(disp + eps) - torch.log(mean + eps)))
#     nb_final = t1 + t2
#     nb_case = nb_final - torch.log(1.0-pi+eps)
#     zero_nb = torch.pow(disp/(disp+mean+eps), disp)
#     zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
#     result = torch.where(torch.le(X, 1e-8), zero_case, nb_case)
        
#     if ridge_lambda > 0:
#         ridge = ridge_lambda*torch.square(pi)
#         result += ridge
#     result = torch.mean(result)
#     return result


###################################### 支撑OPT的sinkhorn函数 ##############################################

def sinkhorn(pred, lambdas, row, col):
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    p = np.power(pred, lambdas)
    
    u = np.ones(num_node)
    v = np.ones(num_class)

    for index in range(1000):
        u = row * np.power(np.dot(p, v), -1)
        u[np.isinf(u)] = -9e-15
        v = col * np.power(np.dot(u, p), -1)
        v[np.isinf(v)] = -9e-15
    u = row * np.power(np.dot(p, v), -1)
    target = np.dot(np.dot(np.diag(u), p), np.diag(v))
    return target