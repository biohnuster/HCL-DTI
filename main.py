import os
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import KFold
import scipy.sparse as sp
from modAL import ModAL
import data
from utils.params import args, params
import utils.util as util

def load_matrix(dataset_path, file_name):
    for ext in ['.txt', '.csv']:
        file_path = os.path.join(dataset_path, file_name + ext)
        if os.path.exists(file_path):
            delimiter = ',' if ext == '.csv' else r'\s+'
            df = pd.read_csv(file_path, delimiter=delimiter, header=None)
            return df.values
    raise FileNotFoundError(f"Neither {file_name}.txt nor {file_name}.csv found in {dataset_path}")

def main(param):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.debug:
        util.set_seed(args.seed)

    dataset_dir = f'dataset/{args.dataset}'
    mat_data_dir = os.path.join(dataset_dir, 'mat_data')
    sim_network_dir = os.path.join(dataset_dir, 'sim_network')
    drug_drug = load_matrix(mat_data_dir, 'mat_drug_drug')
    protein_protein = load_matrix(mat_data_dir, 'mat_protein_protein')
    association = load_matrix(mat_data_dir, 'mat_drug_protein')
    drug_sim = load_matrix(sim_network_dir, 'Sim_mat_drugs')
    protein_sim = load_matrix(sim_network_dir, 'Sim_mat_proteins')
    drug_emb = util.pca_transform(drug_sim, n_components=param['latdim'], device=device)
    protein_emb = util.pca_transform(protein_sim, n_components=param['latdim'], device=device)


    args.num_drug = association.shape[0]
    args.num_protein = association.shape[1]
    association = csr_matrix(association)
    positive_index_tuple = association.nonzero()
    positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
    fold = 5
    kf = KFold(n_splits=fold, shuffle=True, random_state=args.seed)
    metrics0 = []
    metrics1 = []
    metrics2 = []
    metrics3 = []
    metrics4 = []
    metrics5 = []
    for fold_idx, (train_positive_index, test_positive_index) in enumerate(kf.split(positive_index_list)):
        print(f'Fold {fold_idx + 1}')
        train_positive_list = [positive_index_list[i] for i in train_positive_index]
        test_positive_list = [positive_index_list[i] for i in test_positive_index]
        args.query_num = int(len(train_positive_list) / 10)

        train_matrix = lil_matrix(association.shape)
        test_matrix = lil_matrix(association.shape)
        train_durg, train_protein = zip(*train_positive_list)
        test_durg, test_protein = zip(*test_positive_list)

        train_matrix[train_durg, train_protein] = 1

        train_index_list = train_positive_list[:]
        test_index_list = test_positive_list[:]
        for (r, c) in test_positive_list:
            for _ in range(args.negative_sample_times):
                j = np.random.randint(args.num_protein)
                while (r, j) in test_index_list + train_index_list:
                    j = np.random.randint(args.num_protein)
                test_index_list.append((r, j))

        test_label = np.array([association[r, c] for r, c in test_index_list])
        test_drug_index = np.array([x for x, _ in test_index_list])
        test_protein_index = np.array([y for _, y in test_index_list])
        testData = data.Data(test_label, test_drug_index, test_protein_index)

        num_unlabeled_samples = len(train_positive_list) * 20
        unlabeled_tuple_list = util.generate_unlabeled_samples(train_index_list,
                                                               test_index_list,
                                                               association.shape,
                                                               num_unlabeled_samples)

        for i in range(int(len(train_positive_list) / 2)):
            for _ in range(args.negative_sample_times):
                neg_sample = random.choice(unlabeled_tuple_list)
                train_index_list.append(neg_sample)
                unlabeled_tuple_list.remove(neg_sample)

        train_label = np.array([association[r, c] for r, c in train_index_list])
        train_drug_index = np.array([x for x, _ in train_index_list])
        train_protein_index = np.array([y for _, y in train_index_list])
        trainData = data.Data(train_label, train_drug_index, train_protein_index)
        unlabeled_drug = np.array([x for x, _ in unlabeled_tuple_list])
        unlabeled_protein = np.array([y for _, y in unlabeled_tuple_list])
        unlabeled_label = np.zeros((len(unlabeled_tuple_list),), dtype=np.int64)
        unlabeledData = data.UnlabeledData(unlabeled_label, unlabeled_drug, unlabeled_protein)

        sparse_train_matrix = sp.coo_matrix(train_matrix)
        drug_adj_mat = np.where(drug_drug > 0.5, 1, 0)
        protein_adj_mat = np.where(protein_protein > 0.5, 1, 0)
        drug_adj_mat = sp.csr_matrix(drug_adj_mat)
        protein_adj_mat = sp.csr_matrix(protein_adj_mat)
        mat = sp.vstack([sp.hstack([drug_adj_mat, sparse_train_matrix]),
                         sp.hstack([sparse_train_matrix.transpose(), protein_adj_mat])])
        adjMat = util.normalize_adj(mat, device)

        # time_start = time.time()
        Hd = util.build_hyper_graph(sparse_train_matrix, param['k'])
        Hp = util.build_hyper_graph(sparse_train_matrix.transpose(), param['k'])
        drugHypermat = util.normalize_hyper_edge_mat(Hd, device)
        proteinHypermat = util.normalize_hyper_edge_mat(Hp, device)
        # time_end = time.time()
        # time_sum = time_end - time_start
        # print(time_sum)
        data_loader = data.DataHandler(trainData, testData, adjMat, drugHypermat, proteinHypermat, drug_emb,
                                       protein_emb)

        modAL = ModAL(data_loader, unlabeledData, param, device)
        print("round 0")
        metric = modAL.train()
        print(f'fold {fold_idx + 1}, test: acc = {metric[0]:.4f}, precision = {metric[1]:.4f}, '
              f'f1 = {metric[3]:.4f}, TPR = {metric[4]:.4f}, FPR = {metric[5]:.4f},'
              f'AUC = {metric[6]:.4f}, AUPR = {metric[7]:.4f}')
        metrics0.append(metric)

        num_active_learning_rounds = 5
        for r in range(1, num_active_learning_rounds + 1):
            print("round ", r)
            modAL.query_update_data_hybrid()
            m = modAL.train()
            print(f'fold {fold_idx + 1}, test: acc = {m[0]:.4f}, precision = {m[1]:.4f}, '
                  f'f1 = {m[3]:.4f}, TPR = {m[4]:.4f}, FPR = {m[5]:.4f},'
                  f'AUC = {m[6]:.4f}, AUPR = {m[7]:.4f}')

            if r <= 5:
                locals()[f'metrics{r}'].append(m)

    for i in range(6):
        mean_metric = np.mean(locals()[f'metrics{i}'], axis=0)
        print(f'mean{i}: acc = {mean_metric[0]:.4f}, precision = {mean_metric[1]:.4f}, '
              f'recall = {mean_metric[2]:.4f}, f1 = {mean_metric[3]:.4f}, '
              f'TPR = {mean_metric[4]:.4f}, FPR = {mean_metric[5]:.4f},'
              f'AUC = {mean_metric[6]:.4f}, AUPR = {mean_metric[7]:.4f},'
              f'specificity = {mean_metric[8]:.4f}')
    print(param)


if __name__ == '__main__':
    time_start = time.time()
    param = params[args.dataset]
    main(param)
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)