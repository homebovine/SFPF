import torch as tc
import rpy2.robjects as robjects
import numpy as np
from rpy2.robjects import numpy2ri
import pickle
from sklearn import metrics
from sklearn.cluster import KMeans
# import math
import functools
import gc


numpy2ri.activate()

rstring = """
function(range, nord, nbasis, eval){
    library(fda)
    bbasis = create.bspline.basis(rangeval = range, norder = nord, nbasis = nbasis)
    res = eval.basis(eval, bbasis)
    return(res)
}
"""
splinefun = robjects.r(rstring)

rstring1 = """
function(x, y){
prex = seq(0, 1, length.out = 100)
res = predict(smooth.spline(x, y, cv = TRUE), prex)$y
return(res)
}
"""
smoothspline = robjects.r(rstring1)

nbasis = 9
n = 40
d = 60
p = nbasis
dw = 3

rep_time = 500
opt_cluster_mem = np.zeros([rep_time, d])
opt_cluster_size = np.zeros([rep_time, d])
opt_cluster_num = np.zeros([rep_time, 1])
###
est_betad = tc.zeros([rep_time, n * d])
est_beta = tc.zeros([rep_time, n * d])
est_beta0 = tc.zeros([rep_time, n, d])
est_beta0low = tc.zeros([rep_time, n, d])
est_beta0upp = tc.zeros([rep_time, n, d])
est_std0 = tc.zeros([rep_time, n, d])
###
est_etad = tc.zeros([rep_time, dw])
est_eta = tc.zeros([rep_time, dw])
est_etaupp = tc.zeros([rep_time, dw])
est_etalow = tc.zeros([rep_time, dw])

adRI = np.zeros([rep_time, 1])
lambda_tuning = 2 ** tc.Tensor([-7.0])
# 2** tc.Tensor([-3.0, -2.75, -2.5, -2.25, -2.0])
pre_K = ([1, 2, 3, 4, 5, 6, 7, 8])
pre_K1 = tc.Tensor(pre_K)
# lambda_tuning=lambda_tuning/p

d1 = 10
d2 = 20
d3 = 30
dlist = ([d1, d2, d3])

true_beta1 = tc.zeros([rep_time, n, d1])
true_beta2 = tc.zeros([rep_time, n, d2])
true_beta3 = tc.zeros([rep_time, n, d3])
true_z = tc.zeros([rep_time, n, d])
true_y = tc.zeros([rep_time, n, d])
true_x = tc.zeros([rep_time, n, d])
true_w = tc.zeros([rep_time, n, dw])

a = 3.7
maxitr = 1000
d00 = len(lambda_tuning)
d01 = len(pre_K)
cluster_mem_tuning = tc.zeros([d00, d])
cluster_size_tuning = tc.zeros([d00, d])
cluster_num_tuning = tc.zeros([d00, 1])
BIC_est = tc.zeros([d00, d01])
cluster_mem1 = np.zeros([d00, d01, d])
cluster_num1 = tc.zeros([d00, d01])
vG_clu_final = np.zeros([d00, d01, p, d])
BB_clu_final = tc.zeros([d00, d01, n, d])
Beta_temp = tc.zeros([n, d])
# Beta_templow = np.zeros([n, d])
# Beta_tempupp = np.zeros([n, d])
Beta_tuning = tc.zeros([d00, d01, n * d])
Betad_tuning = tc.zeros([d00, d01, n * d])
Betaupp_tuning = tc.zeros([d00, d01, n * d])
Betalow_tuning = tc.zeros([d00, d01, n * d])
std_tuning0 = tc.zeros([d00, d01, n * d])
# std_tuning1 = np.zeros([d00, d01, n * d])
etad_tuning = tc.zeros([d00, d01, dw])
eta_tuning = tc.zeros([d00, d01, dw])
etaupp_tuning = tc.zeros([d00, d01, dw])
etalow_tuning = tc.zeros([d00, d01, dw])

np.random.seed(2020)
Dmatrixres = np.zeros((1, p * d))
for i in range(d - 1):
    Dmatrix = np.zeros((int(p * d * (d - 1) / 2), p * d))
    Dmatrix[(i * p): ((i + 1) * p), (i * p): ((i + 1) * p)] = np.eye(p)
    Dmatrix[(i * p): ((i + 1) * p), ((i + 1) * p): ((i + 2) * p)] = -np.eye(p)
    Dmatrixres = np.vstack((Dmatrixres, Dmatrix[(i * p): ((i + 1) * p), :]))
    for j in range(1, (d - i - 1)):
        Dmatrix[((i + j) * p): (((i + j) + 1) * p), ((i) * p): ((i + 1) * p)] = np.eye(p)
        Dmatrix[((i + j) * p): (((i + j) + 1) * p), ((i + j + 1) * p): ((i + j + 2) * p)] = -np.eye(p)
        Dmatrixres = np.vstack((Dmatrixres, Dmatrix[((i + j) * p): (((i + j) + 1) * p), :]))
Dmatrix = Dmatrixres[1:, :].copy()
Dmatrix = tc.from_numpy(Dmatrix).float()
DDD = Dmatrix.T @ Dmatrix


def estfun(k, Y, features):
    d_start = sum(dlist[0:k])
    d_end = sum(dlist[0:(k + 1)])
    YS = Y[(d_start * n):(d_end * n)]
    fS = features[(d_start * n):(d_end * n), :]
    MMatrix = fS.T @ fS
    # if tc.det(MMatrix) == 0:
    # MMatrix = MMatrix + (tc.rand([p, p])* 0.02 - 0.01)
    est = MMatrix.pinverse() @ fS.T @ YS
    return est


def Aappend(i, features):
    res = features[(i * n): ((i + 1) * n), :].T @ features[(i * n): ((i + 1) * n), :]
    return res


def yXfun(i, features, yMatrix):
    res = features[(i * n):((i + 1) * n), :].T @ yMatrix[(i * n):((i + 1) * n)]
    return res


true_mem = tc.hstack((tc.repeat_interleave(tc.Tensor([0]), d1), tc.repeat_interleave(tc.Tensor([1]), d2),
                      tc.repeat_interleave(tc.Tensor([2]), d3)))

for rep in range(rep_time):
    ###generate model
    Wr = tc.normal(0, 1, (n, (dw - 1)))
    W = tc.hstack((tc.ones((n, 1)), Wr))
    true_w[rep, :, :] = W
    PW = tc.eye(n) - W @ (W.T @ W).inverse() @ W.T
    Z0 = tc.rand((n, d))
    true_z[rep, :, :] = Z0
    ###beta
    betaZ1 = (1 - 2 * Z0[:, :d1]) ** 2
    betaZ2 = - tc.cos(2 * 3.1415926 * Z0[:, d1:(d1 + d2)])
    betaZ3 = - tc.exp(1 + Z0[:, (d1 + d2):(d1 + d2 + d3)])
    betaZ = tc.hstack((betaZ1, betaZ2, betaZ3))

    true_beta1[rep, :, :] = betaZ1
    true_beta2[rep, :, :] = betaZ2
    true_beta3[rep, :, :] = betaZ3
    ###n-by-d matrix X
    X0 = tc.normal(0, 1, (n, d))
    true_x[rep, :, :] = X0
    ### coefficient for W
    eta0 = tc.ones([dw])
    ### error
    err0 = tc.normal(0, 1, (n, d))
    ### Model
    Y0 = betaZ * X0 + tc.reshape(tc.repeat_interleave(W @ eta0.T, d), [n, d]) + err0
    true_y[rep, :, :] = Y0
    ########################################
    Wn = tc.Tensor.repeat(W, (d, 1))
    # print(Wn.shape)
    PWn = tc.eye(n * d) - Wn @ (Wn.T @ Wn).inverse() @ Wn.T
    Y1 = Y0.T.flatten()
    X1 = X0.T.flatten()
    Y = PWn @ Y1
    X = PWn @ X1
    ZMatrix = Z0.T.flatten()
    XMatrix = X
    yMatrix = Y
    res = splinefun(np.percentile(Z0.numpy(), (0, 100)), nord=4, nbasis=nbasis,
                    eval=robjects.vectors.FloatVector(ZMatrix.numpy()))  #
    BZMatrix = np.asarray(res)  ## a nd X p matrix, go n first then d
    BZMatrix = tc.from_numpy(BZMatrix).float()
    features = tc.diag(XMatrix) @ BZMatrix

    estfun1 = functools.partial(estfun, Y=Y, features=features)
    mapres = list(map(estfun1, range(len(dlist))))
    Gamma_int = tc.cat(mapres)
    ovG = tc.reshape(Gamma_int, [len(dlist), p])

    ###BXBX matrix
    Aappend1 = functools.partial(Aappend, features=features)
    mapres = list(map(Aappend1, range(d)))
    BBX = tc.block_diag(*mapres)

    ##BXY matrix
    yXfun1 = functools.partial(yXfun, features=features, yMatrix=yMatrix)
    mapres = list(map(yXfun1, range(d)))
    yXmatrix = tc.cat(mapres)

    del mapres

    ##Dmatrix

    yXmatrixn = yXmatrix / n
    beta = 0.1
    bDDBX = (beta * DDD + BBX / n).inverse()

    print(rep)
    ###no tuning so j is 0
    j = 0
    ### cluster number is fixed so ii=0
    ii = 0
    for kk in range(len(dlist)):
        Index = np.where(true_mem == kk)
        vG_clu_final[j, ii, :, Index] = ovG[kk, :]
    vG_rep_f = np.reshape(np.repeat(vG_clu_final[j, ii, :, :], n, axis=1), [p, d * n])
    vG_rep_f = tc.from_numpy(vG_rep_f).float()
    residual_est = yMatrix - tc.sum(features * vG_rep_f.T, 1)

    betaZ_est = tc.sum(BZMatrix * vG_rep_f.T, 1)
    diff_betaZ = betaZ.T.flatten() - betaZ_est
    Betad_tuning[j, ii, :] = diff_betaZ
    Beta_tuning[j, ii, :] = betaZ_est

    MMatrix = features.T @ features
    est00 = (MMatrix.inverse() @ features.T) @ yMatrix
    HMatrix = features @ MMatrix.inverse() @ features.T
    red1 = yMatrix - HMatrix @ yMatrix
    residual_est0 = (Y0 - (betaZ_est.reshape(d, n)).T * X0).T.flatten()
    # np.matmul(inv(PW), residual_est)
    eta_est = (Wn.T @ Wn).inverse() @ Wn.T @ residual_est0
    etad_tuning[j,] = eta_est - eta0
    residual_final = residual_est0 - Wn @ eta_est

    SSE0 = residual_final @ residual_final.T
    sigma0 = SSE0 / (residual_final.shape[0] - dw - p)
    covb0 = sigma0 * (Wn.T @ Wn).inverse()
    stdest0 = tc.sqrt(tc.diag(covb0))
    eta_tuning[j, ii, :] = eta_est
    etaupp_tuning[j, ii, :] = eta_est + 1.96 * stdest0
    etalow_tuning[j, ii, :] = eta_est - 1.96 * stdest0
    Wnn = tc.hstack((tc.diag(X1) @ BZMatrix, Wn))

    for kk in range(len(dlist)):
        Index = np.asarray(np.where(true_mem == kk))
        Index_flat = []
        for mm in range(Index.shape[1]):
            start = Index[:, mm] * n
            end = (Index[:, mm] + 1) * n
            Index_update = range(start[0], end[0])
            Index_flat = np.append(Index_flat, Index_update)
        Wnn_clu = Wnn[Index_flat, :]
        # covb = (sigma0 * (Wnn_clu.T @ Wnn_clu).inverse())
        covb = (sigma0 * np.linalg.pinv(Wnn_clu.T @ Wnn_clu))
        covb = covb[:p, :p]
        covBest = BZMatrix[Index_flat, :] @ (covb) @ BZMatrix[Index_flat, :].T
        stdest = tc.sqrt(tc.diag(covBest))
        Betaupp_tuning[j, ii, Index_flat] = betaZ_est[Index_flat] + 1.96 * stdest
        Betalow_tuning[j, ii, Index_flat] = betaZ_est[Index_flat] - 1.96 * stdest
        std_tuning0[j, ii, Index_flat] = stdest

    i0 = 0
    i1 = 0

    est_betad[rep, :] = Betad_tuning[i0, i1, :]
    est_beta[rep, :] = Beta_tuning[i0, i1, :]
    est_beta0[rep, :, :] = Beta_tuning[i0, i1, :].reshape(d, n).T
    est_beta0upp[rep, :, :] = Betaupp_tuning[i0, i1, :].reshape(d, n).T
    est_beta0low[rep, :, :] = Betalow_tuning[i0, i1, :].reshape(d, n).T
    est_std0[rep, :, :] = std_tuning0[i0, i1, :].reshape(d, n).T
    # est_std1[rep, :, :] = std_tuning1[i0, i1, :].reshape(d, n).transpose()

    est_etad[rep, :] = etad_tuning[i0, i1, :]
    est_eta[rep, :] = eta_tuning[i0, i1, :]
    est_etaupp[rep, :] = etaupp_tuning[i0, i1, :]
    est_etalow[rep, :] = etalow_tuning[i0, i1, :]

    with open('Setting4_normal_oracle.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(
            [est_betad, est_beta,est_beta0, est_etad, est_eta, est_etaupp,
             est_etalow, true_beta1, true_beta2, true_beta3, true_z, true_y,
             true_x, true_w, est_beta0upp, est_beta0low, est_std0], f)
    # gc.collect()

with open('Setting4_normal_oracle.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(
        [est_betad, est_beta, est_beta0, est_etad, est_eta, est_etaupp,
         est_etalow, true_beta1, true_beta2, true_beta3, true_z, true_y,
         true_x, true_w, est_beta0upp, est_beta0low, est_std0], f)

###print results
with open('Setting4_normal_oracle.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    est_betad, est_beta, est_beta0, est_etad, est_eta, est_etaupp,est_etalow, true_beta1, true_beta2, true_beta3, true_z, true_y,true_x, true_w, est_beta0upp, est_beta0low, est_std0 = pickle.load(f)

n = 40
d = 60
rep_time = 500

norm1 = np.zeros([rep_time, 1])
norm2 = np.zeros([rep_time, 1])
gammad_rep = est_betad
etad_rep = est_etad
for rep in range(rep_time):
    norm1[rep, :] = np.linalg.norm(gammad_rep[rep, :])
    norm2[rep, :] = np.linalg.norm(etad_rep[rep, :])
###SMSE of $\hat(\beta_j$)
print(np.mean(norm1) / np.sqrt(n * d))
###SMSE of $\hat(\eta$)
print(np.mean(norm2) / np.sqrt(3))