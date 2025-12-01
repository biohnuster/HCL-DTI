import random
import numpy as np
import torch.optim
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import data
from utils.params import args
from net import Net
from utils.loss import contrast_loss, calc_reg_loss
from utils.util import metrics


class ModAL:
    def __init__(self, data_loader, unlabeled_data, param, device):
        self.dataLoader = data_loader
        self.device = device
        self.param = param
        self.unlabeled_data_set = unlabeled_data
        self.model = None
        self.optimizer = None
        self.drug_emb = self.dataLoader.get_drug_emb()
        self.protein_emb = self.dataLoader.get_protein_emb()

    def get_unlabeled_data_set(self):
        return self.unlabeled_data_set

    def get_train_dataset(self):
        return self.dataLoader.get_train_data_set()

    def train(self):
        dHyperMat = self.dataLoader.get_drug_hyper_mat()
        pHyperMat = self.dataLoader.get_protein_hyper_mat()
        adjMat = self.dataLoader.get_adj_mat()
        self.model = Net(dHyperMat, pHyperMat, adjMat, self.drug_emb, self.protein_emb, self.param).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param['lr'],
                                          weight_decay=self.param['weight_decay'])
        trainDataSet = self.dataLoader.get_train_data_set()
        trainLoader = DataLoader(trainDataSet, batch_size=args.batch, shuffle=True)
        testDataSet = self.dataLoader.get_test_data_set()
        testLoader = DataLoader(testDataSet, batch_size=args.tstbatch, shuffle=True)
        for epoch in tqdm(range(1, args.epoch+1), ncols=100):
            train_Loss, train_bceLoss, sloss, regloss = self.train_step(trainLoader)

        label, pred_prob = self.predict_probs(testLoader)
        f_accuracy, f_precision, f_recall, f_f1, f_TPR, f_FPR, f_auc, f_aupr, f_specificity = metrics(label, pred_prob)
        return [f_accuracy, f_precision, f_recall, f_f1, f_TPR, f_FPR, f_auc, f_aupr, f_specificity]

    def train_step(self, trainLoader):
        self.model.train()
        criterion = nn.BCELoss()
        gcnEmbedsLst, hyperEmbedsLst, attn_out_drug, attn_out_protein = self.model()
        bceloss, sloss = 0, 0
        for i, (drug, protein, label) in enumerate(trainLoader):
            drug, protein = drug.to(self.device), protein.to(self.device)
            label = label.to(self.device).float()

            drug_embed = attn_out_drug[drug]
            protein_embed = attn_out_protein[protein]
            pred_probs = self.model.predict(drug_embed, protein_embed)
            pred_probs = pred_probs.squeeze(1)
            bceloss += criterion(pred_probs, label)
            for j in range(1, self.param['gnn_layer'] + 1, 1):
                embeds1 = gcnEmbedsLst[j].detach()
                embeds2 = hyperEmbedsLst[j]
                sloss += (contrast_loss(embeds1[:args.num_drug], embeds2[:args.num_drug],
                                        torch.unique(drug), self.param['temp']) +
                          contrast_loss(embeds1[args.num_drug:], embeds2[args.num_drug:],
                                        torch.unique(protein), self.param['temp']))
        sslLoss = sloss * self.param['ssl_reg']
        regloss = calc_reg_loss(self.model) * self.param['reg']
        loss = bceloss + sslLoss + regloss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, bceloss, sslLoss, regloss

    def predict_probs(self, testLoader):
        label_list = []
        pred_probs_list = []
        self.model.eval()
        with torch.no_grad():
            _, _, attn_out_drug, attn_out_protein = self.model()

            for i, (drug, protein, label) in enumerate(testLoader):
                drug, protein = drug.to(self.device), protein.to(self.device)
                label = label.to(self.device).to(torch.long)
                labels = label.detach().cpu()
                drug_embed = attn_out_drug[drug]
                protein_embed = attn_out_protein[protein]
                pred_prob = self.model.predict(drug_embed, protein_embed)
                pred_probs = pred_prob.cpu()
                label_list.append(labels)
                pred_probs_list.append(pred_probs)
        final_label = np.array([label.numpy() for label in label_list])
        final_label = np.squeeze(final_label)
        final_pred_prob = np.array([pred_probs.numpy() for pred_probs in pred_probs_list])
        final_pred_prob = np.squeeze(final_pred_prob, axis=0)
        return final_label, final_pred_prob

    def predict_prob_dropout(self, n_drop=3):
        unlabeled_data = DataLoader(self.unlabeled_data_set, batch_size=args.batch, shuffle=False)
        self.model.train()
        N = len(self.unlabeled_data_set)
        probs = torch.zeros(N, n_drop)
        for i in range(n_drop):

            with torch.no_grad():
                _, _, attn_out_drug, attn_out_protein = self.model()
                for j, (drug, protein, label, idx) in enumerate(unlabeled_data):
                    drug, protein = drug.to(self.device), protein.to(self.device)
                    drug_embed = attn_out_drug[drug]
                    protein_embed = attn_out_protein[protein]
                    pred_prob = self.model.predict(drug_embed, protein_embed)
                    pred_probs = pred_prob.view(-1).cpu()
                    probs[idx, i] = pred_probs

        return probs

    def get_grad_embedding(self):
        unlabeled_data_set = self.get_unlabeled_data_set()
        self.model.eval()
        probs = torch.zeros([len(unlabeled_data_set)])
        grad_embeddings = torch.zeros([len(unlabeled_data_set), args.latdim * 2])
        out = torch.zeros([len(unlabeled_data_set), args.latdim * 2])
        unlabeled_data = DataLoader(unlabeled_data_set, batch_size=args.batch, shuffle=False)

        with torch.no_grad():
            _, _, attn_out_drug, attn_out_protein = self.model()
            for i, (drug, protein, label, idx) in enumerate(unlabeled_data):
                drug, protein = drug.to(self.device), protein.to(self.device)
                drug_embed = attn_out_drug[drug]
                protein_embed = attn_out_protein[protein]
                pred_prob = self.model.predict(drug_embed, protein_embed)
                pred_probs = pred_prob.view(-1).cpu()
                probs[idx] += pred_probs
                embeds = torch.cat((drug_embed, protein_embed), 1).cpu()
                out[idx] += embeds
                pred_idx = (probs > 0.5).float()
                for j in idx:
                    if pred_idx[j]:
                        grad_embeddings[j][:] = out[j] * probs[j]
                    else:
                        grad_embeddings[j][:] = out[j] * (1 - probs[j]) * -1.0
        return grad_embeddings

    def entropy_dropout_sampling(self, n_drop=3):
        probs = self.predict_prob_dropout(n_drop=n_drop)
        mean_probs = probs.mean(dim=1)
        eps = 1e-8
        mean_probs = mean_probs.clamp(eps, 1 - eps)
        log_probs_pos = torch.log(mean_probs)
        log_probs_neg = torch.log(1 - mean_probs)
        uncertainties = -(mean_probs * log_probs_pos + (1 - mean_probs) * log_probs_neg)  # [N]
        top_n_indices = uncertainties.sort(descending=True)[1][:args.query_num]
        return top_n_indices

    def bald_sampling(self, n_drop=3):
        probs = self.predict_prob_dropout(n_drop=n_drop)
        eps = 1e-8
        probs = probs.clamp(eps, 1 - eps)
        entropy_per_dropout = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
        mean_entropy = entropy_per_dropout.mean(dim=1)
        mean_prob = probs.mean(dim=1)
        entropy_of_mean = -mean_prob * torch.log(mean_prob) - (1 - mean_prob) * torch.log(1 - mean_prob)
        bald_scores = entropy_of_mean - mean_entropy
        _, top_indices = torch.topk(bald_scores, k=args.query_num, largest=True, sorted=False)
        return top_indices

    def badge_sampling(self):
        def init_centers(grad_embeddings):
            grad_embeddings = grad_embeddings.numpy()
            ind = np.argmax([np.linalg.norm(s, 2) for s in grad_embeddings])
            mu = [grad_embeddings[ind]]
            indsAll = [ind]
            centInds = [0.] * len(grad_embeddings)
            cent = 0
            while len(mu) < args.query_num:
                if len(mu) == 1:
                    D2 = pairwise_distances(grad_embeddings, mu).ravel().astype(np.float32)
                else:
                    newD = pairwise_distances(grad_embeddings, [mu[-1]]).ravel().astype(np.float32)
                    for i in range(len(grad_embeddings)):
                        if D2[i] > newD[i]:
                            centInds[i] = cent
                            D2[i] = newD[i]

                Ddist = (D2 ** 2) / sum(D2 ** 2)
                customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
                ind = customDist.rvs(size=1)[0]
                while ind in indsAll:
                    ind = customDist.rvs(size=1)[0]
                mu.append(grad_embeddings[ind])
                indsAll.append(ind)
                cent += 1
            return indsAll

        grad_embeddings = self.get_grad_embedding()
        indices = init_centers(grad_embeddings)
        return indices

    def query_update_data_hybrid(self):
        indices = None
        entropy_indices = self.entropy_dropout_sampling()
        bald_indices = self.bald_sampling()
        badge_indices = torch.tensor(self.badge_sampling())
        indices = np.intersect1d(bald_indices, entropy_indices)
        indices = np.intersect1d(badge_indices, indices)
        n_remian = args.query_num - len(indices)
        entropy_indices = entropy_indices[~np.isin(entropy_indices, indices)]
        bald_indices = bald_indices[~np.isin(bald_indices, indices)]
        badge_indices = badge_indices[~np.isin(badge_indices, indices)]
        entropy_ratio = 0.4
        entropy_num = int(n_remian * entropy_ratio)
        bald_ratio = 0.3
        bald_num = int(n_remian * bald_ratio)
        badge_num = n_remian - entropy_num - bald_num
        entropy_indices_new = entropy_indices[:entropy_num]
        badge_indices_new = badge_indices[:badge_num]
        bald_indices_new = bald_indices[:bald_num]
        indices = np.concatenate([entropy_indices_new, badge_indices_new, bald_indices_new, indices])
        query_item = [self.unlabeled_data_set[idx] for idx in indices]
        query_drug, query_protein, query_label, _ = zip(*query_item)
        query_drug = np.array(query_drug)
        query_protein = np.array(query_protein)
        query_label = np.array(query_label)
        queryData = data.Data(query_label, query_drug, query_protein)
        self.dataLoader.update_train_data_set(queryData)
        unlabeled_item = [self.unlabeled_data_set[i] for i in range(len(self.unlabeled_data_set)) if i not in indices]
        unlabeled_drug, unlabeled_protein, unlabeled_label, _ = zip(*unlabeled_item)
        unlabeled_drug = np.array(unlabeled_drug)
        unlabeled_protein = np.array(unlabeled_protein)
        unlabeled_label = np.array(unlabeled_label)
        self.unlabeled_data_set = data.UnlabeledData(unlabeled_label, unlabeled_drug, unlabeled_protein)