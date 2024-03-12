import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from model import STco
from dataset import SKIN, HERDataset
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import pickle
from utils import get_R


def build_loaders_inference(dataset):
    datasets = []
    if dataset=="her2st":
        for i in range(32):
            dataset = HERDataset(train=False, fold=i)
            print(dataset.id2name[0])
            datasets.append(dataset)
    if dataset == "cSCC":
        for i in range(9):
            dataset = HERDataset(train=False, fold=i)
            print(dataset.id2name[0])
            datasets.append(dataset)

    dataset = torch.utils.data.ConcatDataset(datasets)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Finished building loaders")
    return test_loader


def get_image_embeddings(model_path, model, dataset):
    test_loader = build_loaders_inference(dataset)

    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')
        new_key = new_key.replace('well', 'spot')
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    test_image_embeddings = []
    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)

            spot_feature = batch["expression"].cuda()
            x = batch["position"][:, 0].long().cuda()
            y = batch["position"][:, 1].long().cuda()
            centers_x = model.x_embed(x)
            centers_y = model.y_embed(y)
            spot_embeddings.append(model.spot_projection(spot_feature + centers_x + centers_y))
    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings)


def find(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

    return indices.cpu().numpy()


def save_embeddings(model_path, save_path, datasize, dim,  dataset):
    os.makedirs(save_path, exist_ok=True)

    model = STco(spot_embedding=dim, temperature=1.,
                 image_embedding=1024, projection_dim=256).cuda()

    img_embeddings_all, spot_embeddings_all = get_image_embeddings(model_path, model, dataset)

    img_embeddings_all = img_embeddings_all.cpu().numpy()
    spot_embeddings_all = spot_embeddings_all.cpu().numpy()
    print("img_embeddings_all.shape", img_embeddings_all.shape)
    print("spot_embeddings_all.shape", spot_embeddings_all.shape)

    for i in range(len(datasize)):
        index_start = sum(datasize[:i])
        index_end = sum(datasize[:i + 1])
        image_embeddings = img_embeddings_all[index_start:index_end]
        spot_embeddings = spot_embeddings_all[index_start:index_end]
        print("image_embeddings.shape", image_embeddings.shape)
        print("spot_embeddings.shape", spot_embeddings.shape)
        np.save(save_path + "img_embeddings_" + str(i + 1) + ".npy", image_embeddings.T)
        np.save(save_path + "spot_embeddings_" + str(i + 1) + ".npy", spot_embeddings.T)


SAVE_EMBEDDINGS = False
dataset = "her2st"
names = []
if dataset == "her2st":
    slice_num = 32
    names = os.listdir(r".\dataset\Her2st\data/ST-cnts")
    names.sort()
    names = [i[:2] for i in names][1:33]
if dataset == "cSCC":
    slice_num = 12
    patients = ['P2', 'P5', 'P9', 'P10']
    reps = ['rep1', 'rep2', 'rep3']
    for i in patients:
        for j in reps:
            names.append(i + '_ST_' + j)

datasize = [np.load(f"./data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy").shape[1] for
            name in names]


def get_embedding(SAVE_EMBEDDINGS, dataset):
    if SAVE_EMBEDDINGS:
        for fold in range(slice_num): # 12
            save_embeddings(model_path=f"./model_result/{dataset}/{names[fold]}/best_{fold}.pt",
                            save_path=f"./embedding_result/{dataset}/embeddings_{fold}/",
                            datasize=datasize, dim=785, dataset=dataset) # 171

get_embedding(SAVE_EMBEDDINGS, dataset)
spot_expressions = [np.load(f"./data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy")
                    for name in names]
hvg_pcc_list = []
heg_pcc_list = []

for fold in range(slice_num):
    print(f"evaluating: {fold} slice name: {names[fold]} ")

    save_path = f"./embedding_result/her2st/embeddings_{fold}/"
    spot_embeddings = [np.load(save_path + f"spot_embeddings_{i + 1}.npy") for i in range(32)]
    image_embeddings = np.load(save_path + f"img_embeddings_{fold + 1}.npy")

    image_query = image_embeddings
    expression_gt = spot_expressions[fold]
    spot_embeddings = spot_embeddings[:fold] + spot_embeddings[fold + 1:]
    spot_expressions_rest = spot_expressions[:fold] + spot_expressions[fold + 1:]

    spot_key = np.concatenate(spot_embeddings, axis=1)
    expression_key = np.concatenate(spot_expressions_rest, axis=1)

    save_path = f"./her2st_pred_result/{names[fold]}/"
    os.makedirs(save_path, exist_ok=True)
    if image_query.shape[1] != 256:
        image_query = image_query.T
    if expression_gt.shape[0] != image_query.shape[0]:
        expression_gt = expression_gt.T
    if spot_key.shape[1] != 256:
        spot_key = spot_key.T
    if expression_key.shape[0] != spot_key.shape[0]:
        expression_key = expression_key.T

    indices = find(spot_key, image_query, top_k=200)
    spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
    spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
    for i in range(indices.shape[0]):
        a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1)
        reciprocal_of_square_a = np.reciprocal(a ** 2)
        weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
        weights = weights.flatten()
        spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0, weights=weights)
        spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0,
                                                weights=weights)

    np.save(save_path + f"spot_expression_pred_stco_{fold}_{names[fold]}.npy", spot_expression_pred.T)
    true = expression_gt
    pred = spot_expression_pred

    gene_list_path = ".\dataset\Her2st\data/her_hvg_cut_1000.npy"
    gene_list = list(np.load(gene_list_path, allow_pickle=True))
    adata_ture = anndata.AnnData(true)
    adata_pred = anndata.AnnData(pred)

    adata_pred.var_names = gene_list
    adata_ture.var_names = gene_list

    gene_mean_expression = np.mean(adata_ture.X, axis=0)
    top_50_genes_indices = np.argsort(gene_mean_expression)[::-1][:50]
    top_50_genes_names = adata_ture.var_names[top_50_genes_indices]
    top_50_genes_expression = adata_ture[:, top_50_genes_names]
    top_50_genes_pred = adata_pred[:, top_50_genes_names]

    heg_pcc, heg_p = get_R(top_50_genes_pred, top_50_genes_expression)
    hvg_pcc, hvg_p = get_R(adata_pred, adata_ture)
    hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]

    heg_pcc_list.append(np.mean(heg_pcc))
    hvg_pcc_list.append(np.mean(hvg_pcc))

print(f"avg heg pcc: {np.mean(heg_pcc_list):.4f}")
print(f"avg hvg pcc: {np.mean(hvg_pcc_list):.4f}")
