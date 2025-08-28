import numpy as np
#import random

def ti_star(status_nodes,T_BP):
    """Function to compute the ground truth vector of times of infection

    Args:
        status_nodes (array): Array of shape (T+1) x N contaning the ground-truth states of the system from t=0 to t=T

    Returns:
        ti (array): Array of size N containing the ground-truth vector of times of infection
    """
    N = len(status_nodes[0])
    ti = np.zeros(N)
    for i in range(N):
        t_inf = np.nonzero(status_nodes[:, i] == 1)[0]
        if len(t_inf) == 0:
            ti[i] = T_BP
        else:
            ti[i] = t_inf[0] - 1
    return ti


def ti_inferred(B):
    """Function to compute the MMSE-estimation of the times of infection

    Args:
        B (array): Array of shape N x (T+2) containing the beliefs estimated by BP

    Returns:
        ti_infer (array): Array of size N contaning the MMSE-estimation of the times of infection
    """
    ti_infer = np.array(
        [np.array(
            [(t - 1) * bt for t, bt in enumerate(b)]
            ).sum() for b in B]
    )
    return ti_infer

def ti_random(B):
    """Function to compute the RND-estimation of the times of infection

    Args:
        B (array): Array of shape N x (T+2) containing the beliefs estimated by BP

    Returns:
        ti_rnd (array): Array of size N contaning the RND-estimation of the times of infection
    """
    N = B.shape[0]
    T = len(B[0])-2
    b_mean = B.mean(axis=0)
    ti_rnd =  np.array(
        [ t * b_mean[i]  for i,t  in enumerate(range(-1,T+1))]
    ).sum()
    return np.full(N,ti_rnd)


def OV(conf1, conf2):
    """Function to compute the overlap between two arrays of the same shape

    Args:
        conf1 (array): First array
        conf2 (array): Second array

    Returns:
        ov (float): Overlap
    """
    ov = np.mean(conf1 == conf2)
    return ov


def MOV(Mt):
    """Function to compute the MMO-Mean overlap, given the array of marginals

    Args:
        Mt (array): Array of marginals

    Returns:
        mov (float): MMO-Mean overlap
    """
    M0 = np.maximum(Mt[0],Mt[1])
    mov = np.mean(np.maximum(M0, Mt[2]))
    return mov


def OV_rnd(conf, Mt):
    """Function to compute the RND-overlap, given the array of marginals

    Args:
        conf (array): Array of configurations
        Mt (array): Array of marginals

    Returns:
        mov_rnd (float): RND-overlap
    """
    x = np.argmax(np.mean(Mt, axis=1))
    ov_rnd = np.mean(conf == x)
    return ov_rnd


def MOV_rnd(Mt):
    """Function to compute the RND-Mean overlap, given the array of marginals

    Args:
        Mt (array): Array of marginals

    Returns:
        mov_rnd (float): RND-Mean overlap
    """
    m1 = np.maximum(np.mean(Mt[0]), np.mean(Mt[1]))
    mov_rnd = np.maximum(m1, np.mean(Mt[2]))
    return mov_rnd


def SE(ti_star, ti_inferred):
    """Function to compute the SE, given the ground truth vector of times of infection and an estimation of it

    Args:
        ti_star (array): Array of size N containing the ground-truth vector of times of infection
        ti_inferred (array):  Array of size N contaning an estimation of the times of infection

    Returns:
        se (float): SE
    """
    se = np.array([(t - ti_inferred[i]) ** 2 for i, t in enumerate(ti_star)]).mean()
    return se


def MSE(B, ti_inferred):
    """Function to compute the MSE, given the ground truth vector of times of infection and an estimation of it

    Args:
        B (array): Array of shape N x (T+2) containing the beliefs estimated by BP
        ti_inferred (array):  Array of size N contaning an estimation of the times of infection

    Returns:
        mse (float): MSE
    """
    mse = np.array(
        [
            np.array([b * (ti - (t - 1)) ** 2 for t, b in enumerate(B[i])]).sum()
            for i, ti in enumerate(ti_inferred)
        ]
    ).mean()
    return mse