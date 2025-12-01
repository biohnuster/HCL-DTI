import random
import warnings
import numpy as np
import torch
import torch as t
import scipy.sparse as sp
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

from utils.params import args


def set_seed(seed):
    print('seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True


def generate_unlabeled_samples(train_index_list, test_index_list, association_shape, num_samples):
    row_index, col_index = np.indices(association_shape)
    all_index_tuple_list = list(zip(row_index.flatten(), col_index.flatten()))
    labeled_set = set(train_index_list + test_index_list)
    unlabeled_tuple_list = []
    while len(unlabeled_tuple_list) < num_samples:
        item = random.choice(all_index_tuple_list)
        if item not in labeled_set:
            unlabeled_tuple_list.append(item)
            labeled_set.add(item)
    return unlabeled_tuple_list


def build_hyper_graph(matrix, k):

    A_mid = matrix.dot(matrix.T)
    HB_list = []
    E = sp.csr_matrix(np.eye(matrix.shape[0]))
    HB_list.append(E)
    if k > 1:
        for i in range(2, k + 1):
            A = np.power(A_mid, i)
            A[A > 1] = 1
            HB_list.append(A)
    H = HB_list[0].dot(matrix)
    for i in range(1, len(HB_list)):
        H += HB_list[i].dot(matrix)

    return H


def normalize_adj(mat, device):
    if args.dataset == 'data_mei':
        mat = (mat + sp.eye(mat.shape[0]))
    degree = np.array(mat.sum(axis=-1))
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    dInvSqrtMat = sp.diags(dInvSqrt)
    mat = mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
    idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = t.from_numpy(mat.data.astype(np.float32))
    shape = t.Size(mat.shape)
    return t.sparse_coo_tensor(idxs, vals, shape).to(device)


def normalize_hyper_edge_mat(mat, device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dDegree = np.array(mat.sum(axis=1))
        dDegree = np.maximum(dDegree, 1e-12)
        dInvSqrt = np.reshape(np.power(dDegree, -0.5), [-1])
        dInvSqrt = np.where(dInvSqrt >= 1e+12, 0.0, dInvSqrt)
        dInvSqrtMat = sp.diags(dInvSqrt)
        hDegree = np.array(mat.sum(axis=0))
        hDegree = np.maximum(hDegree, 1e-12)
        hInv = np.reshape(np.power(hDegree, -1), [-1])
        hInv = np.where(hInv >= 1e+12, 0.0, hInv)
        hInvMat = sp.diags(hInv)

        result = dInvSqrtMat @ mat @ hInvMat @ mat.transpose() @ dInvSqrtMat
        result_coo = result.tocoo()
        idxs = t.from_numpy(np.vstack([result_coo.row, result_coo.col]).astype(np.int64))
        vals = t.from_numpy(result_coo.data.astype(np.float32))
        shape = t.Size(result_coo.shape)
        return t.sparse_coo_tensor(idxs, vals, shape).to(device)


def pca_transform(emb, n_components=128, device=None):
    if isinstance(emb, np.ndarray):
        tensor = torch.from_numpy(emb).float()
    else:
        tensor = emb
    tensor = tensor.to(device)
    batch_size = tensor.size(0)
    x = tensor.reshape(batch_size, -1)
    mean = torch.mean(x, dim=0, keepdim=True)
    x = x - mean
    U, S, V = torch.pca_lowrank(x, q=n_components)
    output = torch.matmul(x, V[:, :n_components])
    return output


def metrics(label, pred_prob):
    preds = (pred_prob > 0.5).astype(int)
    cm = confusion_matrix(label, preds)
    TN, FP, FN, TP = cm.ravel()
    accuracy = accuracy_score(label, preds)
    precision = precision_score(label, preds, zero_division=1)
    recall = recall_score(label, preds)
    f1 = f1_score(label, preds)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    AUC = roc_auc_score(label, pred_prob)
    AUPR = average_precision_score(label, pred_prob)
    Specificity = TN / (TN + FP)
    return accuracy, precision, recall, f1, TPR, FPR, AUC, AUPR, Specificity
