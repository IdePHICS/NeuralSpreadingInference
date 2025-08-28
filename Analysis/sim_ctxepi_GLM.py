import numpy as np
import networkx as nx
import random, sys, time, lzma, pickle, argparse, warnings
import pandas as pd
from pathlib import Path

from Analysis.gen import *
from XZtoDF_AMP_GLM import data_to_dict
from Modules._AMP_BP_Epi_GLM import *
from Modules import fg
from Modules._AMP_BP_Epi_GLM import BP_AMPstep

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute marginals using BPEpI")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/check/",
        help="saving directory for data",
    )
    parser.add_argument(
        "--save_DF_dir",
        type=str,
        default="../data/check/",
        help="saving directory for data frames",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="data",
        help="name of the file in which the marginals will be saved",
    )
    parser.add_argument("--graph", type=str, default="rrg", help="Type of random graph")
    parser.add_argument(
        "--N", type=int, default=10000, help="Number of individuals", nargs="+"
    )
    parser.add_argument("--d", type=int, default=3, help="degree of RRG", nargs="+")
    parser.add_argument("--lam", type=float, default=1.0, help="lambda", nargs="+")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="fraction of sources, pass as multiple arguments, e.g. 0.1 0.3 0.5",
        nargs="+",
    )
    group_o = parser.add_mutually_exclusive_group(required=True)
    group_o.add_argument(
        "--n_obs",
        type=int,
        default=[-1],
        help="number of observations, pass as multiple arguments, e.g. 2 4 8",
        nargs="+",
    )
    group_o.add_argument(
        "--rho",
        type=float,
        default=[-1],
        help="fraction of observations, pass as multiple arguments, e.g. 0.1 0.3 0.5",
        nargs="+",
    )
    parser.add_argument("--nsim", type=int, default=1, help="number of simulations")
    parser.add_argument(
        "--print_it",
        action="store_true",
        help="If false, save just the marginals at convergence. If true, save marginals every 'iter_space' iterations of BP",
    )
    parser.add_argument(
        "--iter_space",
        type=int,
        default=100,
        help="Space between saved iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the number generators",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Tolerance of the difference between marginals to stop BP ",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=500,
        help="Max number of iterations of the algorithm",
    )
    group_ot = parser.add_mutually_exclusive_group(required=True)
    group_ot.add_argument(
        "--sens",
        action="store_const",
        const="sensors",
        dest="obs_type",
        help="Snapshot observation",
    )
    group_ot.add_argument(
        "--snap",
        action="store_const",
        const="snapshot",
        dest="obs_type",
        help="Snapshot observation",
    )
    parser.add_argument(
        "--snap_time",
        type=float,
        default=-1,
        help="Time at which taking the snapshot. Random by default",
    )
    parser.add_argument(
        "--T_max",
        type=int,
        default=500,
        help="Max number of timesteps of the simulation",
    )

    parser.add_argument(
        "--T_BP_max",
        type=int,
        default=100,
        help="Maximum time of infectionn that can be inferred by BP",
    )

    group_d = parser.add_mutually_exclusive_group(required=True)
    group_d.add_argument(
        "--Delta",
        type=int,
        default=-1,
        help="Lenght of mask array, filled with ones",
    )
    group_d.add_argument(
        "--mu",
        type=float,
        default=-1,
        help="Value of recovery probability of the SIR model to simulate through dSIR",
    )
    group_d.add_argument(
        "--mask",
        type=float,
        default=1.0,
        help="Mask array",
        nargs="+",
    )
    group_d.add_argument(
        "--SI",
        action="store_true",
        help="Set the mask array in order to simulate SI model",
    )
    parser.add_argument(
        "--tol2",
        type=float,
        default=1e-3,
        help="Tolerance of the difference between marginals to stop BP for the second part ",
    )
    parser.add_argument(
        "--it_max",
        type=int,
        default=50000,
        help="Max number of iterations of the algorithm, after the first threshold ",
    )
    parser.add_argument(
        "--save_marginals",
        action="store_true",
        help="If false, save just the Data Frame. If true, save also the marginals found by BP",
    )
    parser.add_argument(
        "--SIR_sim",
        action="store_true",
        help="If true, simulates a conventional SIR model",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.0,
        help="Damping factor for the BP iterations. Needs to be smaller than 0.5. Default: 0",
    )
    group_i = parser.add_mutually_exclusive_group(required=True)
    group_i.add_argument(
        "--unif_init",
        action="store_true",
        help="Start BP from uniform marginals",
    )
    group_i.add_argument(
        "--rnd_init",
        action="store_true",
        help="Start BP from 'uninformed' solution",
    )
    group_i.add_argument(
        "--inf_init",
        action="store_true",
        help="Start BP from 'informed' solution",
    )
    group_i.add_argument(
        "--rnd_inf_init",
        action="store_true",
        help="Run BP from both rnd and inf initializations",
    )
    parser.add_argument(
        "--infer_up_to",
        type=float,
        default=-1,
        help="If -1, infer up to the end of the epidemic. If different from -1, gives the maximum time of infection that can be inferred by BP",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.,
        help="Constant in the output function ",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=10000,
        help="Dimension of the features ",
    )
    parser.add_argument(
        '--uRademacher', 
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Type of u-variables. Specify if Rademacher (otherwise Gaussian) ",
    )

    return parser.parse_args()

def print_iter(err, it):
    print(f"err:{err[0]:.2}, err:{err[1]:.2}, it:{it}")

def add_init_obs(obs_list, ground_truth):
    """
    Adds the initial observations to the list of observations.
    """
    for i in range(len(ground_truth)):
        obs_list.append((i, ground_truth[i], 0))
    return obs_list

def BPloop_AMP(f, list_obs, n_iter, tol, print_it, iter_space, tol2, it_max, init, damp, F, a0, v0, go_0, eta_0, nu_0, c, uRad, s0):
    # Initialization
    M =F.shape[1]
    A = np.sum(go_0**2)/M
    B = a0*A+np.dot(go_0, F) 
    #free_entropy_AMP_real_inf = loglikelihood_AMP(f, F, a0, go_0, v0, A, B, uRad) # real free entropy of the informed initialization
    print_space = 1
    if ((init == 1) or (init == 2)):
        for it in range(it_max):
            err_max, err_mean, x0, a0, v0, go_0, eta_0, nu_0 = BP_AMPstep(f,0., a0, v0, go_0, eta_0, nu_0, F, c, uRad)
            if it % print_space == 0:
                print_iter([err_max, err_mean], it)
            if err_mean < tol:
                break
        if err_mean > tol:
            warnings.warn("Warning... Initialization is not converging")
    print("ended init")  
    if init == 2: 
        list_obs = add_init_obs(list_obs, s0)
    f.reset_obs(list_obs)
    x=x0
    a=a0
    v=v0
    go=go_0
    eta = eta_0
    nu = nu_0
    err_space = 10
    e_max = np.nan
    e_ave = np.nan
    err_list = []
    # BP iteration
    if print_it:
        marg_list = [f.marginals()]
        mess_list = [f.get_messages()]
        it_list = [0]
        e_list = [e_ave]
        logL_list = [f.loglikelihood()]
    for it in np.arange(1, n_iter + 1):
        a_old = a
        v_old = v
        e_max, e_ave, x, a, v, go, eta, nu = BP_AMPstep(f,0.01,a,v,go,eta, nu, F,c, uRad)
        if print_it and ((it % iter_space == 0) or ((e_ave < (1-0.01)*tol) and (np.mean((a-a_old)**2) < (1-0.01)*tol) and (np.mean((v-v_old)**2) < (1-0.01)*tol)) or (it == n_iter)):
            marg_list.append(f.marginals())
            mess_list.append(f.get_messages())
            it_list.append(it)
            e_list.append(e_ave)
            logL_list.append(f.loglikelihood())
        if it % err_space == 0:
            err_list.append([it, e_max, e_ave])
        if it % print_space == 0:
            print(it,e_ave,np.mean((a-a_old)**2),np.mean((v-v_old)**2),np.mean(1*(x<0) == s0),np.mean(np.maximum((1-x)/2,1-(1-x)/2)))
        if ( ((e_ave < (1-0.01)*tol) and (np.mean((a-a_old)**2) < (1-0.01)*tol) and (np.mean((v-v_old)**2) < (1-0.01)*tol))):
            break
    if e_ave < tol2:
        while ( ((e_ave > (1-0.01)*tol) or (np.mean((a-a_old)**2) > (1-0.01)*tol) or (np.mean((v-v_old)**2) > (1-0.01)*tol))):
            it = it + 1
            a_old = a
            v_old = v
            e_max, e_ave, x, a, v, go, eta, nu = BP_AMPstep(f,0.01,a,v,go,eta,nu,F,c, uRad)
            if print_it and ((it % iter_space == 0) or ((e_ave < (1-0.01)*tol) and (np.mean((a-a_old)**2) < (1-0.01)*tol) and (np.mean((v-v_old)**2) < (1-0.01)*tol)) or (it == it_max)  ):
                marg_list.append(f.marginals())
                mess_list.append(f.get_messages())
                it_list.append(it)
                e_list.append(e_ave)
                logL_list.append(f.loglikelihood())
            if it % err_space == 0:
                err_list.append([it, e_max, e_ave])
            if it % print_space == 0:
                print(it,e_ave,np.mean((a-a_old)**2),np.mean((v-v_old)**2),np.mean(1*(x<0) == s0),np.mean(np.maximum((1-x)/2,1-(1-x)/2)))
            if it == n_iter * 2:
                break
    if e_ave > tol: 
        while  ( ((e_ave > (1-damp)*tol) or (np.mean((a-a_old)**2) > (1-damp)*tol) or (np.mean((v-v_old)**2) > (1-damp)*tol))):
            it = it + 1
            a_old = a
            v_old = v
            e_max, e_ave, x, a, v, go, eta,nu = BP_AMPstep(f,damp,a,v,go,eta,nu,F,c, uRad)
            if print_it and ((it % iter_space == 0) or ((e_ave < (1-damp)*tol) and (np.mean((a-a_old)**2) < (1-damp)*tol) and (np.mean((v-v_old)**2) < (1-damp)*tol)) or (it == it_max)):
                marg_list.append(f.marginals())
                mess_list.append(f.get_messages())
                it_list.append(it)
                e_list.append(e_ave)
                logL_list.append(f.loglikelihood())
            if it % err_space == 0:
                err_list.append([it, e_max, e_ave])
            if it % print_space == 0:
                print(it,e_ave,np.mean((a-a_old)**2),np.mean((v-v_old)**2),np.mean(1*(x<0) == s0),np.mean(np.maximum((1-x)/2,1-(1-x)/2)))
            if it == n_iter * 3:
                break
        while  ( ((e_ave > (1-2*damp)*tol) or (np.mean((a-a_old)**2) > (1-2*damp)*tol) or (np.mean((v-v_old)**2) > (1-2*damp)*tol))):
            it = it + 1
            e_max, e_ave, x, a, v, go, eta,nu = BP_AMPstep(f,2*damp,a,v,go,eta,nu,F,c, uRad)
            if print_it and ((it % iter_space == 0) or ((e_ave < (1-2*damp)*tol) and (np.mean((a-a_old)**2) < (1-2*damp)*tol) and (np.mean((v-v_old)**2) < (1-2*damp)*tol)) or (it == it_max)):
                marg_list.append(f.marginals())
                mess_list.append(f.get_messages())
                it_list.append(it)
                e_list.append(e_ave)
                logL_list.append(f.loglikelihood())
            if it % err_space == 0:
                err_list.append([it, e_max, e_ave])
            if it % print_space == 0:
                print(it,e_ave,np.mean((a-a_old)**2),np.mean((v-v_old)**2),np.mean(1*(x<0) == s0),np.mean(np.maximum((1-x)/2,1-(1-x)/2)))
            if it == it_max:
                break

    if not print_it:
        marg_list = [f.marginals()]
        mess_list = [f.get_messages()]
        it_list = [it]
        e_list = [e_ave]
        logL_list = [f.loglikelihood()]

    if it != it_max:
        err_list.append([it, e_max, e_ave])

    M =F.shape[1]
    A = np.sum(go**2)/M
    B = a*A+np.dot(go, F)   

    free_entropy_AMP = loglikelihood_AMP(f, F, a, go, v, A, B,c, uRad)
    free_entropy_BP = f.loglikelihood_BP() 
        

    return mess_list, marg_list, e_list, it_list, logL_list, err_list, free_entropy_AMP, free_entropy_BP

def main():
    args = parse_arguments()
    print("arguments:")
    print(args)

    save_dir = args.save_dir
    save_DF_dir = args.save_DF_dir
    path_save = Path(save_dir)
    if not path_save.exists():
        warnings.warn("SAVING FOLDER DOES NOT EXIST")
    graph = args.graph
    if graph == "rrg":

        def generate_graph(N, d):
            return nx.random_regular_graph(n=N, d=d)

    elif graph == "tree":

        def generate_graph(N, d):
            return nx.full_rary_tree(r=d, n=N)
        
    elif graph == "ER":

        def generate_graph(N, d):
            G = nx.erdos_renyi_graph(n=N, p=d/(N-1))
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            ii =1
            for i in range(len(G.nodes)):
                if i not in G.nodes : 
                    G = nx.relabel_nodes(G, {list(G.nodes)[-ii]:i}, copy=True)
                    ii += 1
            return G

    else:
        warnings.warn("GRAPH TYPE NOT ALLOWED")
    N_table = args.N
    d_table = args.d
    lam_table = args.lam
    if len(np.shape(args.delta)) == 1:
        sources_table = args.delta
        s_type = "delta"
    else:
        sources_table = args.n_sources
        s_type = "n_sources"
    if len(np.shape(args.rho)) == 1:
        obs_table = args.rho
        o_type = "rho"
    else:
        obs_table = args.n_obs
        o_type = "n_obs"
    n_sim = args.nsim
    print_it = args.print_it
    iter_space = args.iter_space
    seed = args.seed  # setting seed everywhere for reproducibility TBD
    random.seed(seed)
    np.random.seed(seed)
    tol = args.tol
    n_iter = args.n_iter
    T_max = args.T_max
    T_BP_max = args.T_BP_max
    mu = 0
    if args.SI == True:
        mask = ["SI"]
        Delta = T_max + 1
        mask_type = "SI"
    elif args.Delta != -1:
        mask = [1] * args.Delta
        Delta = args.Delta
        mask_type = "dSIR_one"
    elif args.mu != -1:
        mu = args.mu
        mask = [(1 - mu) ** i for i in range(T_max + 1)]
        Delta = T_max + 1
        mask_type = "dSIR_exp"
    else:
        mask = args.mask
        Delta = len(mask)
        mask_type = "dSIR_custom"
    tol2 = args.tol2
    it_max = args.it_max
    save_marginals = args.save_marginals
    SIR_sim = args.SIR_sim
    damp = min(args.damping,0.495)
    infer_up_to = int(np.ceil(args.infer_up_to))
    M = args.M
    c = args.c
    uRad = args.uRademacher

    dict_list = []
    t1 = time.time()
    t2 = time.time()
    for i_N, N in enumerate(N_table):
        for i_d, d in enumerate(d_table):
            for i_l, lam in enumerate(lam_table):
                for i_S, S in enumerate(sources_table):
                    for i_Mobs, Mobs in enumerate(obs_table):
                        for sim in range(n_sim):
                            if len(np.shape(args.delta)) == 0:
                                pseed = S / N
                            else:
                                pseed = S
                            G = generate_graph(N=N, d=d)
                            for (u, v) in G.edges():
                                G.edges[u, v]["lambda"] = lam
                            N = G.number_of_nodes()
                            if SIR_sim == True:
                                warnings.warn(
                                    "TO BE IMPLEMENTED!"
                                )
                                sys.exit()
                            else:
                                us = generate_us(M, wRademacher=uRad)
                                F = generate_F(N, M)
                                s0 = 1*((np.dot(F, us)+c)<0)
                                ground_truth = simulate_one_detSIR_list(
                                    G, s0=s0, mask=mask, T_max=T_max
                                )
                            if ( (len(ground_truth) > T_BP_max ) & (infer_up_to == -1) ):
                                warnings.warn(
                                    "The simulation exeeds the maximum time limit!"
                                )
                                sys.exit()
                            varInit = N**-0.5
                            T = len(ground_truth)-1
                            #x0=s0
                            #goU_0, _ = compute_go(bp_fg_U.compute_nu(), om, V,c)

                            
                            
                            if (infer_up_to == -1) : T_BP = T
                            else : T_BP = infer_up_to
                            contacts = generate_contacts(G, T_BP, lam)
                            if args.obs_type == "sensors":
                                list_obs = generate_sensors_obs(
                                    ground_truth, o_type=o_type, M=Mobs, T_max=T_BP
                                )
                                list_obs_all = generate_sensors_obs(
                                    ground_truth, o_type="rho", M=1, T_max=T_BP
                                )
                                fS = np.mean(ground_truth[-1] == 0)
                                fI = np.mean(ground_truth[-1] == 1)
                                TO = T_BP
                            else:
                                list_obs, fS, fI, TO = generate_snapshot_obs(
                                    ground_truth,
                                    o_type=o_type,
                                    M=Mobs,
                                    snap_time=args.snap_time,
                                    i_u_t = infer_up_to
                                )
                                list_obs_all = generate_sensors_obs(
                                    ground_truth, o_type="rho", M=1, T_max=T_BP
                                )
                            f = {}
                            if (args.rnd_init == True):
                                f = fg.FactorGraph(
                                    N=N,
                                    T=T_BP,
                                    contacts=contacts,
                                    obs=[],
                                    delta=np.full(N,0.5),
                                    mask=mask,
                                    mask_type=mask_type,
                                )
                                a0 = np.random.normal(0, 1, M)*varInit
                                v0 = np.ones(M)
                                V = np.mean(v0)
                                om = np.dot(F, a0)
                                nu_0 = np.full(N,0.5)#f.compute_nu()
                                eta_0 = compute_eta(om, V, c)
                                go_0, _ = compute_go(nu_0, om, V,c)
                                (
                                    _,
                                    marg_list_rnd,
                                    eR_list,
                                    itR_list,
                                    logLR_list,
                                    errR_list,
                                    free_entropy_AMP,
                                    free_entropy_BP
                                ) = BPloop_AMP(
                                    f,
                                    list_obs,
                                    n_iter,
                                    tol,
                                    print_it,
                                    iter_space,
                                    tol2,
                                    it_max,
                                    init=1,
                                    damp=damp, 
                                    F = F,
                                    a0 = a0, 
                                    v0 = v0, 
                                    go_0 = go_0,
                                    eta_0 = eta_0,
                                    nu_0 = nu_0,
                                    c = c,
                                    uRad = uRad,
                                    s0 = ground_truth[0]
                                )
                            elif (args.inf_init == True):
                                f = fg.FactorGraph(
                                    N=N,
                                    T=T_BP,
                                    contacts=contacts,
                                    obs=list_obs_all,
                                    delta=pseed,
                                    mask=mask,
                                    mask_type=mask_type,
                                )
                                a0 = us
                                v0 = np.ones(M)*varInit# NB: Diminishing the variance makes BP diverge!! ???
                                V = np.mean(v0)
                                om = np.dot(F, a0)
                                nu_0 = np.full(N,0.5)
                                for o in list_obs_all:
                                    nu_0[o[0]] = 1 - ground_truth[0][o[0]] 
                                #nu_0 = f.compute_nu(
                                eta_0 = compute_eta(om, V, c)
                                go_0, _ = compute_go(nu_0, om, V,c)
                                (
                                    _,
                                    marg_list_inf,
                                    eI_list,
                                    itI_list,
                                    logLI_list,
                                    errI_list,
                                    free_entropy_AMP,
                                    free_entropy_BP
                                ) = BPloop_AMP(
                                    f,
                                    list_obs,
                                    n_iter,
                                    tol,
                                    print_it,
                                    iter_space,
                                    tol2,
                                    it_max,
                                    init=2,
                                    damp=damp,
                                    F = F,
                                    a0 = a0, 
                                    v0 = v0, 
                                    go_0 = go_0,
                                    eta_0 = eta_0,
                                    nu_0 = nu_0,
                                    c = c,
                                    uRad = uRad,
                                    s0 = ground_truth[0]
                                )

                            elif args.unif_init == True:
                                f = fg.FactorGraph(
                                    N=N,
                                    T=T_BP,
                                    contacts=contacts,
                                    obs=list_obs,
                                    delta=np.full(N,pseed),
                                    mask=mask,
                                    mask_type=mask_type,
                                )
                                a0 = np.zeros(M)#np.random.normal(0, 1, M)*varInit
                                v0 = np.ones(M)
                                V = np.mean(v0)
                                om = np.dot(F, a0)
                                nu_0 = np.full(N,0.5)#f.compute_nu()
                                eta_0 = compute_eta(om, V, c)
                                go_0 = np.zeros(N)# , _ = compute_go(f.compute_nu(), om, V,c)
                                (
                                    _,
                                    marg_list_unif,
                                    eU_list,
                                    itU_list,
                                    logLU_list,
                                    errU_list,
                                    free_entropy_AMP,
                                    free_entropy_BP
                                ) = BPloop_AMP(
                                    f,
                                    list_obs,
                                    n_iter,
                                    tol,
                                    print_it,
                                    iter_space,
                                    tol2,
                                    it_max,
                                    init=0,
                                    damp=damp,
                                    F = F,
                                    a0 = a0, 
                                    v0 = v0, 
                                    go_0 = go_0,
                                    eta_0 = eta_0,
                                    nu_0 = nu_0,
                                    c = c,
                                    uRad = uRad,
                                    s0 = ground_truth[0]
                                )
                            elif args.rnd_inf_init == True:
                                f = fg.FactorGraph(
                                    N=N,
                                    T=T_BP,
                                    contacts=contacts,
                                    obs=[],
                                    delta=np.full(N,0.5),
                                    mask=mask,
                                    mask_type=mask_type,
                                )
                                a0 = np.random.normal(0, 1, M)*varInit
                                v0 = np.ones(M)
                                V = np.mean(v0)
                                om = np.dot(F, a0)
                                nu_0 = np.full(N,0.5)#f.compute_nu()
                                eta_0 = compute_eta(om, V, c)
                                go_0, _ = compute_go(nu_0, om, V,c)
                                (
                                    mess_list_rnd,
                                    marg_list_rnd,
                                    eR_list,
                                    itR_list,
                                    logLR_list,
                                    errR_list,
                                    free_entropy_AMP_rnd,
                                    free_entropy_BP_rnd
                                ) = BPloop_AMP(
                                    f,
                                    list_obs,
                                    n_iter,
                                    tol,
                                    print_it,
                                    iter_space,
                                    tol2,
                                    it_max,
                                    init=1,
                                    damp=damp, 
                                    F = F,
                                    a0 = a0, 
                                    v0 = v0, 
                                    go_0 = go_0,
                                    eta_0 = eta_0,
                                    nu_0 = nu_0,
                                    c = c,
                                    uRad = uRad,
                                    s0 = ground_truth[0]
                                )
                                print("BP free entropy (random init):", free_entropy_BP_rnd)
                                print("AMP free entropy (random init):", free_entropy_AMP_rnd)

                                f = fg.FactorGraph(
                                    N=N,
                                    T=T_BP,
                                    contacts=contacts,
                                    obs=list_obs_all,
                                    delta=np.full(N,0.5),
                                    mask=mask,
                                    mask_type=mask_type,
                                )
                                a0 = us
                                v0 = np.ones(M)*varInit# NB: Diminishing the variance makes BP diverge!! ???
                                V = np.mean(v0)
                                om = np.dot(F, a0)
                                nu_0 = np.full(N,0.5)
                                for o in range(N):
                                    nu_0[o] = 1 - ground_truth[0][o]
                                #nu_0 = f.compute_nu()
                                eta_0 = compute_eta(om, V, c)
                                go_0, _ = compute_go(nu_0, om, V,c)
                                (
                                    mess_list_inf,
                                    marg_list_inf,
                                    eI_list,
                                    itI_list,
                                    logLI_list,
                                    errI_list,
                                    free_entropy_AMP_inf,
                                    free_entropy_BP_inf
                                ) = BPloop_AMP(
                                    f,
                                    list_obs,
                                    n_iter,
                                    tol,
                                    print_it,
                                    iter_space,
                                    tol2,
                                    it_max,
                                    init=2,
                                    damp=damp,
                                    F = F,
                                    a0 = a0, 
                                    v0 = v0, 
                                    go_0 = go_0,
                                    eta_0 = eta_0,
                                    nu_0 = nu_0,
                                    c = c,
                                    uRad = uRad,
                                    s0 = ground_truth[0]
                                )
                                
                                print("BP free entropy (informed init):", free_entropy_BP_inf)
                                print("AMP free entropy (informed init):", free_entropy_AMP_inf)
                                print("Free entropy difference (random minus informed):", free_entropy_BP_rnd + free_entropy_AMP_rnd - (free_entropy_BP_inf + free_entropy_AMP_inf))


                            print(
                                f"\r N: {i_N+1}/{len(N_table)} - d: {i_d+1}/{len(d_table)} - lam: {i_l+1}/{len(lam_table)} - S: {i_S+1}/{len(sources_table)} - M: {i_Mobs+1}/{len(obs_table)} - sim: {sim+1}/{n_sim} - time = {time.time()-t2:.2f} s - total time = {time.time()-t1:.0f} s"
                            )
                            t2 = time.time()
                            timestr = (
                                "_"
                                + time.strftime("%Y%m%d-%H%M%S")
                                + "_"
                                + str(time.time())[-6:]
                            )
                            if print_it:
                                file_name = "IT_" + args.file_name + timestr + f"_{seed}.xz"
                            else:
                                file_name = args.file_name + timestr + f"_{seed}.xz"


                            saveObj1 = (
                                graph,
                                N,
                                d,
                                lam,
                                s_type,
                                S,
                                o_type,
                                Mobs,
                                iter_space,
                                seed,
                                tol,
                                n_iter,
                                args.obs_type,
                                args.snap_time,
                                T_max,
                                mask,
                                mask_type,
                                mu,
                                tol2,
                                it_max,
                                ground_truth,
                                G,
                                T,
                                list_obs,
                                fS,
                                fI,
                                TO,
                                T_BP,
                                infer_up_to,
                                Delta,
                                damp,
                                c,
                                uRad,
                                M
                            )
                            if args.rnd_init == True:
                                init_type = 0
                                total_free_entropy = free_entropy_BP + free_entropy_AMP
                                saveObj2 = (
                                    marg_list_rnd,
                                    eR_list,
                                    itR_list,
                                    logLR_list,
                                    errR_list,
                                    free_entropy_BP,
                                    free_entropy_AMP,
                                    total_free_entropy
                                )
                            elif args.inf_init == True:
                                init_type = 1
                                total_free_entropy = free_entropy_BP + free_entropy_AMP
                                saveObj2 = (
                                    marg_list_inf,
                                    eI_list,
                                    itI_list,
                                    logLI_list,
                                    errI_list,
                                    free_entropy_BP,
                                    free_entropy_AMP,
                                    total_free_entropy
                                )
                            elif args.rnd_inf_init == True:
                                init_type = 2
                                total_free_entropy_rnd = free_entropy_BP_rnd + free_entropy_AMP_rnd
                                total_free_entropy_inf = free_entropy_BP_inf + free_entropy_AMP_inf
                                #if (len(mess_list_rnd) < len(mess_list_inf)):
                                #    l = len(mess_list_rnd)
                                #    SE_mess = [np.mean((mess_list_rnd[i]-mess_list_inf[i])**2) for i in range(l)]
                                #    for i_se in np.arange(l,len(mess_list_inf)):
                                #        SE_mess.append(np.mean((mess_list_rnd[l-1]-mess_list_inf[i_se])**2))
                                #else:
                                #    l = len(mess_list_inf)
                                #    SE_mess = [np.mean((mess_list_rnd[i]-mess_list_inf[i])**2) for i in range(l)]
                                #    for i_se in np.arange(l,len(mess_list_rnd)):
                                #        SE_mess.append(np.mean((mess_list_inf[l-1]-mess_list_rnd[i_se])**2))

                                saveObj2 = (
                                    #SE_mess,
                                    marg_list_rnd,
                                    marg_list_inf,
                                    eR_list,
                                    eI_list,
                                    itR_list,
                                    itI_list,
                                    logLR_list,
                                    logLI_list,
                                    errR_list,
                                    errI_list,
                                    free_entropy_BP_rnd,
                                    free_entropy_AMP_rnd,
                                    total_free_entropy_rnd,
                                    free_entropy_BP_inf,
                                    free_entropy_AMP_inf,
                                    total_free_entropy_inf,
                                )
                            elif args.unif_init == True:
                                init_type = 3
                                total_free_entropy = free_entropy_BP + free_entropy_AMP
                                saveObj2 = (
                                    marg_list_unif,
                                    eU_list,
                                    itU_list,
                                    logLU_list,
                                    errU_list,
                                    free_entropy_BP,
                                    free_entropy_AMP,
                                    total_free_entropy
                                )

                            if save_marginals:
                                with lzma.open(save_dir + file_name, "wb") as f:
                                    pickle.dump([saveObj1, saveObj2], f)
                            dict_list = dict_list + data_to_dict(
                                saveObj1, saveObj2, init_type
                            )
    data_frame = pd.DataFrame(dict_list)
    timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(time.time())[-6:]
    file_name = "DF_CtxtEpiGLM_d" + str(d_table[0]) + graph + "_N" + str(N) + "_lam" + str(lam) + "_del" + str(args.delta)+ "_rho" + str(args.rho) + "_c" + str(c) + "_M" + str(M)+ "_uRad" + str(1*uRad)+ "_init" + str(init_type)+ "_seed" + str(seed) + ".xz"
    if print_it:
        file_name = "IT_" + file_name
    with lzma.open(save_DF_dir + file_name, "wb") as f:
        pickle.dump(data_frame, f)


if __name__ == "__main__":
    main()
