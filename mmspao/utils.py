import torch, scipy, scanpy, numpy, random
from typing import Literal
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

def set_random_seed(seed: int = 0):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def getX(adata, modality: Literal["rna", "atac", "protein", "metabolite"]):
    adata.var_names_make_unique()
    if modality in ['rna', 'atac']:
        scanpy.pp.filter_genes(adata, min_cells=10)
        scanpy.pp.log1p(adata)

        if scipy.sparse.issparse(adata.X):
            return adata.X.toarray()
        else:
            return adata.X

    elif modality=='protein':
        adata.X = numpy.apply_along_axis(protein_norm, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else numpy.array(adata.X)))
        return adata.X     

    elif modality=='metabolite':
        scanpy.pp.log1p(adata)
        if scipy.sparse.issparse(adata.X):
            return adata.X.toarray()
        else:
            return adata.X

def protein_norm(x):
    s = numpy.sum(numpy.log1p(x[x > 0]))
    exp = numpy.exp(s / len(x))
    return numpy.log1p(x / exp)

def cal_dist(X1, sparse: bool = False, neighbors: int = 100):
    D = pairwise_distances(X1)
    sig = numpy.median(D[numpy.triu_indices_from(D, k=1)])
    if not sparse:
        a1 = numpy.exp(-1*(D**2)/(2*(sig**2)))
        return a1
    else:
        dist1 = kneighbors_graph(X1, n_neighbors = neighbors, mode='distance')
        dist1.data = numpy.exp(-1*(dist1.data**2)/(2*(sig**2)))
        dist1.eliminate_zeros()
        return dist1
