import torch, numpy, scanpy
from sklearn.preprocessing import StandardScaler
from .utils import cal_dist
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix
from itertools import combinations
from .nets import KAN
from typing import Union, List, Tuple, Literal

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

class mmspao(torch.nn.Module):
    def __init__(self, features, sel_views: Union[List[int], Literal["all"]]='all', combination: Union[List[Tuple[int, int]], Literal["all"]]='all', sparse: bool=False, neighbors: Union[int, None] = None, device: Literal["cpu", 'cuda']='cpu'):
        super(mmspao, self).__init__()
        self.device = device
        self.num_views = len(features) # numbers of modalities
        self.sparse = sparse # get from scipy.sparse.issparse(adata.X)
        if neighbors is None and self.sparse:
            neighbors = 100
        adj, pcs = [], []
        for i in tqdm(features, desc='perform PCA for each modality'):
            i = StandardScaler().fit_transform(i)
            adj.append(cal_dist(i, sparse = self.sparse, neighbors=neighbors))
            pcs.append(PCA(128).fit_transform(i) if i.shape[1] > 128 else i)
        self.pcs = [torch.Tensor(i).to(self.device) for i in pcs] 
        if not self.sparse:
            self.adj = [torch.Tensor(i).to(self.device) for i in adj]
        else:
            indices, values, shape = [], [], []
            for i in tqdm(adj, desc='sparse adj for each modality'):
                i = coo_matrix(i)
                indices.append(torch.LongTensor(numpy.vstack((i.row, i.col))))
                values.append(torch.FloatTensor(i.data))
                shape.append(torch.Size(i.shape))
            self.adj = [torch.sparse_coo_tensor(indices[i], values[i], shape[i]).to(self.device) for i in range(len(adj))]

        if sel_views == 'all':
            self.sel_views = list(range(len(self.pcs)))
        else:
            self.sel_views = sel_views   
        if combination == 'all':
            self.combinations = list(combinations(list(range(len(self.pcs))),2))
        else:
            self.combinations = combination        

    def train(self, epochs: int=1000):
        self.models = [KAN(pca_shape = self.pcs[i].shape[1], latent_reps = 64).to(self.device) for i in range(len(self.pcs))]
        def graph_emb(mod_adj, cur_emb):
            if not self.sparse:
                return (torch.triu(torch.cdist(cur_emb, cur_emb))*torch.triu(mod_adj)).mean()
            else:
                indices0 = mod_adj.coalesce().indices()[0]
                indices1 = mod_adj.coalesce().indices()[1]
                rows = cur_emb[indices0]
                cols = cur_emb[indices1]
                norm_dist = torch.norm(rows - cols, dim=1)
                return (norm_dist*mod_adj.coalesce().values()).mean()

        for i in range(self.num_views):
            self.models[i].train()
            optimizer = torch.optim.Adam(self.models[i].parameters(), lr=1e-3)
            pbar = tqdm(range(epochs), desc='Training KAN for modality ' + str(i+1))          
            for epoch in pbar:
                optimizer.zero_grad()
                gen = self.models[i](self.pcs[i])
                latent = self.models[i].get_latent(self.pcs[i])
                loss1 = torch.nn.MSELoss()(self.pcs[i], gen)
                loss2 = graph_emb(self.adj[i], latent)
                loss = loss1 + loss2
                pbar.set_postfix({'gen_loss' : '{:.3f}'.format(loss1),
                                  'sc_loss': '{:.3f}'.format(loss2),
                                  'loss': '{:.3f}'.format(loss)})
                loss.backward()
                optimizer.step()

        for i in range(self.num_views):
            self.models[i].eval()
        eval_embs = [self.models[i].get_latent(self.pcs[i]) for i in range(self.num_views)]
        if self.combinations is not None:
            exterior = [eval_embs[i][:, :, None]*eval_embs[j][:, None, :] for i, j in self.combinations]
            exterior = [i.reshape(i.shape[0], -1) for i in exterior]
            exterior = [torch.matmul(i, torch.pca_lowrank(i, q=64)[2]) for i in exterior]
        eval_embs = [eval_embs[i] for i in self.sel_views]
        eval_embs = [StandardScaler().fit_transform(i.cpu().detach().numpy()) for i in eval_embs]
        eval_embs = numpy.concatenate(eval_embs, 1)
        if self.combinations is not None:
            exterior = [StandardScaler().fit_transform(i.cpu().detach().numpy()) for i in exterior]
            exterior = numpy.concatenate(exterior, 1)
            emb = numpy.concatenate((eval_embs, exterior), 1)
        else:
            emb = eval_embs
        self.emb = emb
    
    def cluster(self, adata, n_domains: int, n_neighbors: int=50, end: float=2.5):
        assert end > 0.1
        adata.obsm['emb'] = self.emb
        scanpy.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='emb')
        pbar = tqdm(sorted(numpy.arange(0.1, end, 0.01), reverse=True), desc='Leiden clustering')
        for res in pbar:
            scanpy.tl.leiden(adata, random_state=0, resolution=res, flavor="igraph", n_iterations=2, directed=False)
            if len(adata.obs['leiden'].unique()) == n_domains:
                print(f"Found resolution: {res:.2f} for {n_domains} domains")
                return res
        return 1.0

    def modularity(self, adata):
        import igraph as ig
        G_coo = adata.obsp['connectivities'].tocoo()
        edges = list(zip(G_coo.row.tolist(), G_coo.col.tolist()))
        weights = G_coo.data.tolist()
        g = ig.Graph(n=adata.n_obs, edges=edges, directed=False)
        g.es['weight'] = weights
        membership = adata.obs['leiden'].astype(int).tolist()
        mod = g.modularity(membership, weights='weight')
        return mod
    
    def moran(self, adata, k=6):
        import libpysal
        from esda.moran import Moran
        coords = adata.obsm['spatial']
        labels = adata.obs['leiden'].astype(int).values
        w = libpysal.weights.KNN.from_array(coords, k=k)
        w.transform = 'r'
        mi = Moran(labels, w)
        return mi.I, mi.p_sim
    
    def contiguity(self, adata, k=6):
        from sklearn.neighbors import NearestNeighbors
        labels = adata.obs['leiden'].astype(int).values
        coords = adata.obsm['spatial']
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        same = 0
        total = 0
        for i, neigh in enumerate(indices):
            for j in neigh[1:]:  # skip self
                same += (labels[i] == labels[j])
                total += 1
        return same / total
