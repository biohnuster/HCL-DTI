from torch.utils.data import Dataset, ConcatDataset


class DataHandler:
    def __init__(self, trainData, testData, adjMat, dHypermat, pHyperMat, drug_emb, protein_emb):
        self.trainDataSet = trainData
        self.testDataSet = testData
        self.adjMat = adjMat
        self.drugHypermat = dHypermat
        self.proteinHypermat = pHyperMat
        self.drug_emb = drug_emb
        self.protein_emb = protein_emb

    def get_drug_emb(self):
        return self.drug_emb

    def get_protein_emb(self):
        return self.protein_emb

    def get_train_data_set(self):
        return self.trainDataSet

    def get_test_data_set(self):
        return self.testDataSet

    def update_train_data_set(self, queryDataSet):
        self.trainDataSet = ConcatDataset([self.trainDataSet, queryDataSet])

    def get_adj_mat(self):
        return self.adjMat

    def get_protein_hyper_mat(self):
        return self.proteinHypermat

    def get_drug_hyper_mat(self):
        return self.drugHypermat


class Data(Dataset):
    def __init__(self, labels, d_indices, p_indices):
        self.labels = labels
        self.d_indices = d_indices
        self.p_indices = p_indices

    def __len__(self):
        return len(self.d_indices)

    def __getitem__(self, idx):
        return self.d_indices[idx], self.p_indices[idx], self.labels[idx]


class UnlabeledData(Dataset):
    def __init__(self, labels, d_indices, p_indices):
        self.labels = labels
        self.d_indices = d_indices
        self.p_indices = p_indices

    def __len__(self):
        return len(self.d_indices)

    def __getitem__(self, idx):
        return self.d_indices[idx], self.p_indices[idx], self.labels[idx], idx
