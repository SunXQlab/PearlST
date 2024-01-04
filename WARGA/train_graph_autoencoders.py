# -*- coding: utf-8 -*-
"""
Created on Nov 1 2023

@author: Haiyun Wang
"""



import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.autograd as autograd
from torch.autograd import Variable
from WARGA.model import GCNModelAE, Regularizer
from WARGA.optimizer import loss_function1
from WARGA.utils import preprocess_graph
from sklearn.cluster import KMeans
from WARGA.utils import graph_alpha
from sklearn.metrics import adjusted_rand_score


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Cited from Improved Training of Wasserstein GANs
# https://github.com/igul222/improved_wgan_training
def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates,
                              inputs=interpolates,
                              grad_outputs=fake,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 0.01) ** 2).mean()
    return gradient_penalty


def pretrain_gae(args, adata):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    concat_X = adata.obsm['augment_data']
    print("Using {} dataset".format(args.dataset_name))
    features = concat_X.astype(np.float32)
    n_nodes, feat_dim = features.shape
    features = torch.tensor(features).to(device)
    adj = graph_alpha(adata.obsm['spatial'], n_neighbors=10)
    adj = adj.astype(np.float32)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    # Some preprocessing
    adj_norm = preprocess_graph(adj_orig)
    adj_norm = adj_norm.to(device)

    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    adj_label = adj_label.to(device)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    regularizer = Regularizer(in_channels=args.reg_in_channels, hidden_dim1=args.reg_hidden1,
                              hidden_dim2=args.reg_hidden2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    regularizer_optimizer = optim.Adam(regularizer.parameters(), lr=args.reglr)

    for epoch in range(1, args.epochs+1):
        model.train()
        regularizer.train()
        # Generate embeddings
        predicted_labels_prob, emb = model(features, adj_norm)

        # Wasserstein Regularizer
        for i in range(5):
            f_z = regularizer(emb).to(device)
            r = torch.normal(0.0, 1.0, [n_nodes, args.hidden2]).to(device)
            f_r = regularizer(r)
            # add the gradient penalty to objective function
            gradient_penalty = compute_gradient_penalty(regularizer, r, emb)
            reg_loss = - f_r.mean() + f_z.mean() + args.gp_lambda * gradient_penalty
            regularizer_optimizer.zero_grad()
            reg_loss.backward(retain_graph=True)
            regularizer_optimizer.step()

        # GAE Update
        f_z = regularizer(emb)
        generator_loss = -f_z.mean()
        loss = loss_function1(preds=predicted_labels_prob, labels=adj_label,
                              norm=norm, pos_weight=torch.tensor(pos_weight))
        loss = loss + generator_loss
        optimizer.zero_grad()
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss))
            np_emb = emb.cpu().detach().numpy()

    np_emb = emb.cpu().detach().numpy()
    adata.obsm["PearlST_embed"] = np_emb
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed).fit(np_emb)
    predict_labels = kmeans.predict(np_emb)
    adata.obs['pred_label'] = [str(x) for x in predict_labels]
    adata_obs = adata.obs.dropna()
    ari = adjusted_rand_score(adata_obs['ground_truth'], adata_obs['pred_label'])
    print("ARI=", "{:.5f}".format(ari))
    return adata, model

