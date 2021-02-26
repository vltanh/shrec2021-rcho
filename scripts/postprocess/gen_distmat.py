import argparse

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """
    q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
    q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
    g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
    k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
    Returns:
        final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
    """
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(
        1. * original_dist/np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1+1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(
                np.around(k1/2.))+1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + \
                np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def dotprod_dist(x, y):
    d = x.dot(y.T)
    d = (d - d.min(1)[:, np.newaxis]) / (d.max(1) - d.min(1))[:, np.newaxis]
    return 1 - d


def dist_func(mode):
    factory = {
        'euclidean': euclidean_distances,
        'cosine': cosine_distances,
        'dotprod': dotprod_dist,
    }
    if mode in factory:
        return factory[mode]
    else:
        raise Exception('Invalid mode.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query',
                        type=str,
                        help='path to the query embeddings')
    parser.add_argument('-g', '--gallery',
                        type=str,
                        help='path to the gallery embeddings')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='output file name and directory')
    parser.add_argument('-m', '--mode',
                        type=str,
                        help='distance [euclidean|cosine|dotprod]')
    parser.add_argument('-fmt', '--format',
                        type=str,
                        default='%10.6f',
                        help='printing format of each float')
    parser.add_argument('-r', '--rerank',
                        action='store_true',
                        help='whether or not to re-rank')
    return parser.parse_args()


args = parse_args()

q = np.load(args.query)
g = np.load(args.gallery)
dist = dist_func(args.mode)

if args.rerank:
    q_g = dist(q, g)
    q_q = dist(q, q)
    g_g = dist(g, g)

    dist_mat = re_ranking(q_g, q_q, g_g)
else:
    dist_mat = dist(q, g)
np.savetxt(args.output, dist_mat, fmt=args.format)
