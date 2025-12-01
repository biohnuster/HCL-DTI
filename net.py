import torch
from torch import nn
import torch.nn.functional as F
from utils.params import args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, dHyperMat, pHyperMat, adjMat, drug_emb, protein_emb, params):
        super(Net, self).__init__()
        self.gnn_layer = params['gnn_layer']
        self.hlambda = params['hlambda']
        self.hidden_dim = params['latdim']

        self.drug_embeds = nn.Parameter(drug_emb)
        self.protein_embeds = nn.Parameter(protein_emb)
        self.weights = nn.Parameter(torch.ones(self.gnn_layer+1)/(self.gnn_layer+1))
        self.Q_drug = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K_drug = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V_drug = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.beta_drug = nn.Parameter(torch.ones(1))

        self.Q_protein = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K_protein = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V_protein = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.beta_protein = nn.Parameter(torch.ones(1))

        self.dHyperMat = dHyperMat
        self.pHyperMat = pHyperMat
        self.adjMat = adjMat

        self.gcnLayer = GCN()
        self.d_hgcnLayer = HGCN()
        self.p_hgcnLayer = HGCN()
        self.mlp = MLP(self.hidden_dim)

    def forward(self):
        embeds = torch.cat([self.drug_embeds, self.protein_embeds], dim=0)
        embedsLst = [embeds]
        gcnEmbedsLst = [embeds]
        hyperEmbedsLst = [embeds]

        for i in range(self.gnn_layer):
            gcnEmbeds = self.gcnLayer(self.adjMat, embedsLst[i].to(device))
            hyperMEmbeds = self.d_hgcnLayer(self.dHyperMat, embedsLst[i][:args.num_drug].to(device))
            hyperDEmbeds = self.p_hgcnLayer(self.pHyperMat, embedsLst[i][args.num_drug:].to(device))
            hyperEmbeds = torch.cat([hyperMEmbeds, hyperDEmbeds], dim=0)

            gcnEmbedsLst.append(gcnEmbeds)
            hyperEmbedsLst.append(hyperEmbeds)
            embedsLst.append(self.hlambda * hyperEmbeds + gcnEmbeds)

        embedsLst = [embed.to(device) for embed in embedsLst]
        embeds_stack = torch.stack(embedsLst, dim=0)
        weights = self.weights.view(-1, 1, 1)

        embeds = (embeds_stack * weights).sum(dim=0)

        dEmbeds = embeds[:args.num_drug]
        pEmbeds = embeds[args.num_drug:]
        drug_proj = dEmbeds
        protein_proj = pEmbeds
        assert drug_proj.shape == (args.num_drug, self.hidden_dim) and protein_proj.shape == (
            args.num_protein, self.hidden_dim)

        Q_drug = self.Q_drug(drug_proj)
        K_drug = self.K_drug(drug_proj)
        V_drug = self.V_drug(drug_proj)
        # assert Q_drug.shape == K_drug.shape == V_drug.shape == (args.num_drug, args.latdim)

        Q_protein = self.Q_protein(protein_proj)
        K_protein = self.K_protein(protein_proj)
        V_protein = self.V_protein(protein_proj)
        # assert Q_protein.shape == K_protein.shape == V_protein.shape == (args.num_protein, args.latdim)

        attn_drug = Q_drug @ K_drug.T
        # assert attn_drug.shape == (args.num_drug, args.num_drug)
        attn_drug = torch.softmax(attn_drug, dim=-1)
        attn_out_drug = self.beta_drug * (attn_drug @ V_drug) + drug_proj
        # assert attn_out_drug.shape == (args.num_drug, args.latdim)

        attn_protein = Q_protein @ K_protein.T
        # assert attn_protein.shape == (args.num_protein, args.num_protein)
        attn_protein = torch.softmax(attn_protein, dim=-1)
        attn_out_protein = self.beta_protein * (attn_protein @ V_protein) + protein_proj
        # assert attn_out_protein.shape == (args.num_protein, args.latdim)

        return gcnEmbedsLst, hyperEmbedsLst, attn_out_drug, attn_out_protein

    def predict(self, drug, protein):
        pred_probs = self.mlp(drug, protein)
        return pred_probs



class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, mat, embeds):
        embedding = torch.matmul(mat, embeds)
        embedding = self.leakyRelu(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class HGCN(nn.Module):
    def __init__(self):
        super(HGCN, self).__init__()
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, hyperMat, embeds):
        result = hyperMat @ embeds
        result = self.leakyRelu(result)
        result += embeds
        result = F.normalize(result, p=2, dim=1)
        return result


class MLP(nn.Module):
    def __init__(self, dim):
        super(MLP, self).__init__()
        self.embedding_size = dim * 2
        self.drop_rate = 0.35
        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction.apply(init_weights)

    def forward(self, dEmbeds, pEmbeds):
        embeds = torch.cat((dEmbeds, pEmbeds), 1)
        pred_probs = self.mlp_prediction(embeds)
        return pred_probs



