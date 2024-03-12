import numpy as np
from os import name
from PIL import Image
import numpy as np
import scanpy as sc, anndata as ad
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pandas as pd
import anndata as ann
import os
import scprep as scp
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def get_R(data1, data2, dim=1, func=pearsonr):
    adata1 = data1.X
    adata2 = data2.X
    r1, p1 = [], []
    for g in range(data1.shape[dim]):
        if dim == 1:
            r, pv = func(adata1[:, g], adata2[:, g])
        elif dim == 0:
            r, pv = func(adata1[g, :], adata2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1


Image.MAX_IMAGE_PIXELS = 933120000
BCELL = ['CD19', 'CD79A', 'CD79B', "MS4A1"]
TUMOR = ['FASN']
CD4T = ['CD4']
CD8T = ['CD8A', 'CD8B']
DC = ['CLIC2', 'CLEC10A', 'CD1B', 'CD1A', 'CD1E']
MDC = ['LAMP3']
CMM = ['BRAF', 'KRAS']

IG = {'B_cell': BCELL, 'Tumor': TUMOR, 'CD4+T_cell': CD4T, 'CD8+T_cell': CD8T, 'Dendritic_cell': DC,
      'Mature_dendritic_cell': MDC, 'Cutaneous_Malignant_Melanoma': CMM}

MARKERS = []
for i in IG.values():
    MARKERS += i
# print(MARKERS)
LYM = {'B_cell': BCELL, 'CD4+T_cell': CD4T, 'CD8+T_celll': CD8T}


def read_tiff(path):
    Image.MAX_IMAGE_PIXELS = 933120000
    im = Image.open(path)
    imarray = np.array(im)
    return im


def preprocess(adata, n_keep=1000, include=LYM, g=True):
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    if g:
        b = list(np.load('data/skin_hvg_cut_1000.npy', allow_pickle=True))
        b.remove('WHSC1L1')
        b.remove('AC013461.1')
        b.remove('MLLT4')
        adata = adata[:, b]
    elif include:
        exp = np.zeros((adata.X.shape[0], len(include)))
        for n, (i, v) in enumerate(include.items()):
            tmp = adata[:, v].X
            tmp = np.mean(tmp, 1).flatten()
            exp[:, n] = tmp
        adata = adata[:, :len(include)]
        adata.X = exp
        adata.var_names = list(include.keys())
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_keep, subset=True)
    c = adata.obsm['spatial']
    scaler = preprocessing.StandardScaler().fit(c)
    c = scaler.transform(c)
    adata.obsm['position_norm'] = c
    return adata


def comp_umap(adata):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="clusters")
    return adata


def comp_tsne_km(adata, k=10):
    sc.pp.pca(adata)
    sc.tl.tsne(adata)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(adata.obsm['X_pca'])
    adata.obs["kmeans"] = kmeans.labels_.astype(str)
    return adata


def co_embed(a, b, k=10):
    a.obs['tag'] = 'Truth'
    b.obs['tag'] = 'Pred'
    adata = ad.concat([a, b])
    sc.pp.pca(adata)
    sc.tl.tsne(adata)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(adata.obsm["X_pca"])
    adata.obs['kmeans'] = kmeans.labels_.astype(str)
    return adata


def build_adata(name='H1'):
    cnt_dir = '../data/her2st/data/ST-cnts'
    img_dir = '../data/her2st/data/ST-imgs'
    pos_dir = '../data/her2st/data/ST-spotfiles'

    pre = img_dir + '/' + name[0] + '/' + name
    fig_name = os.listdir(pre)[0]
    path = pre + '/' + fig_name
    im = Image.open(path)

    path = cnt_dir + '/' + name + '.tsv.gz'
    cnt = pd.read_csv(path, sep='\t', index_col=0)

    path = pos_dir + '/' + name + '_selection.tsv'
    df = pd.read_csv(path, sep='\t')

    x = df['x'].values
    y = df['y'].values
    id = []
    for i in range(len(x)):
        id.append(str(x[i]) + 'x' + str(y[i]))
    df['id'] = id

    meta = cnt.join((df.set_index('id')))

    gene_list = list(np.load('data/her_g_list.npy'))
    adata = ann.AnnData(scp.transform.log(scp.normalize.library_size_normalize(meta[gene_list].values)))
    adata.var_names = gene_list
    adata.obsm['spatial'] = np.floor(meta[['pixel_x', 'pixel_y']].values).astype(int)
    return adata, im


def get_data(dataset='bc3', n_keep=1000, include=LYM, g=True):
    if dataset == 'bc1':
        adata = sc.datasets.visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_1', include_hires_tiff=True)
        adata = preprocess(adata, n_keep, include, g)
        img_path = adata.uns["spatial"]['V1_Breast_Cancer_Block_A_Section_1']["metadata"]["source_image_path"]
    elif dataset == 'bc2':
        adata = sc.datasets.visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_2', include_hires_tiff=True)
        adata = preprocess(adata, n_keep, include, g)
        img_path = adata.uns["spatial"]['V1_Breast_Cancer_Block_A_Section_2']["metadata"]["source_image_path"]
    else:
        adata = sc.datasets.visium_sge(sample_id=dataset, include_hires_tiff=True)
        adata = preprocess(adata, n_keep, include, g)
        img_path = adata.uns["spatial"][dataset]["metadata"]["source_image_path"]

    return adata, img_path


if __name__ == '__main__':
    adata, img_path = get_data()
    print(adata.X.toarray())