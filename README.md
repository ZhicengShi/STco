# Spatial gene expression prediction from histology images with STco
### Zhiceng Shi, Changmiao Wang, Wen Wen Min*
## Introduction
In this study, we introduce a novel approach: STco, a multi-modal deep learning method that operates within a contrastive learning framework.STco learns a multimodal embedding space from H\&E images, spot gene expression data, and spot positional information. Specifically, the image is passed through an image encoder to capture visual features, while the spot's gene expression data along with its positional encoding is input to the Spot Encoder to capture fused features incorporating spatial information. Contrastive learning is then applied to the obtained visual features and fused features, maximizing the cosine similarity of embeddings for truly paired images and gene expressions, while minimizing the similarity for incorrectly paired embeddings. To predict gene expression from an image, the test image data is fed into the image encoder to extract its visual features. Subsequently, the cosine similarity is computed between the obtained visual features and the features of $N$ spots (consistent with the training process). The top k spot features with the highest similarity scores are selected, and their corresponding ground truth gene expressions are weightedly aggregated to infer the gene expression of the test image. Our method leverages spatial transcriptomics datasets obtained from two distinct tumors using the 10X Genomics platform : these include human HER2-positive breast cancer (HER2+) and human cutaneous squamous cell carcinoma (cSCC) . The empirical results of our study underscore the enhanced performance of STco in accurately predicting gene expression profiles based on histological images, surpassing the capabilities of existing methods.

![(Variational)](WorkFlow.png)

## System environment
Required package:
- PyTorch >= 2.1.0
- scanpy >= 1.8
- python >=3.9

## Datasets

 -  human HER2-positive breast tumor ST data https://github.com/almaan/her2st/.
 -  human cutaneous squamous cell carcinoma 10x Visium data (GSE144240).

## STco pipeline
- 1. Please run the script download.sh in the folder data;
- 2. Run the command line git clone https:/github.com/almaan/her2st.git in the dir data;
- 3. Run train.py
  4. Run pred.py




