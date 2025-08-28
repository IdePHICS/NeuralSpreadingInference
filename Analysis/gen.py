import numpy as np
import random
import networkx as nx
import math
#import sib

def simulate_one_detSIR_list(G, s0, mask = ["SI"], T_max=100):
    """Function to simulate an epidemic using the deterministic-SIR model

    Args:
        G (nx.graph): Graph representing the contact network
        s0: list of initial states of the nodes
        mask (list): if it is equal to ["SI"], the function simulates an SI model, otherwise the i-th element of the
            list (between 0 and 1) represents the infectivity of the nodes at i timesteps after the infection
        T_max (int): maximum allowed time of simulation

    Returns:
        status_nodes (list): array of shape (T+1) x N containing the state of all the nodes from time 0 to time T
    """
    N=G.number_of_nodes()
    s0 = s0.tolist()
    #s0  = [1 if s0_minusplus[i] == -1. else 0 for i in range(N)]
    # Generate the sources
    status_nodes = []
    flag_s=0
    flag_m = 1
    flag_si = 0
    flag_sir = 0
    flag_dsir = 0
    if mask == ["SI"] : 
        mask = [1]
        flag_m = 0
        flag_si = 1
    elif len(mask) == T_max+1 : flag_sir = 1
    else: flag_dsir = 1
    counter = [mask.copy() for _ in range(N)]
    coeff_lam = np.ones(N)
    for i in range(N):
        if s0[i] == 1 : 
            coeff_lam[i]=counter[i][0]
            if flag_m : counter[i].pop(0)
            flag_s = 1        

    status_nodes.append(np.array(s0))
    # Generate the epidemics
    st = np.copy(s0)
    while ((flag_dsir==1 and 1 in status_nodes[-1]) or ( flag_si==1 and 0 in status_nodes[-1]) or (flag_sir==1 and (0 in status_nodes[-1]) and (len(status_nodes) < T_max))) and (len(status_nodes) <= T_max):
        st_minus_1 = np.copy(st)
        coeff_minus_1 = coeff_lam.copy()
        for i in range(N):
            if st[i] == 1 :
                if not counter[i]: st[i]=2
                else :
                    coeff_lam[i]=counter[i][0]
                    if flag_m : counter[i].pop(0)
            elif st[i] == 0 :
                for j in nx.neighbors(G,i) :
                    if (st_minus_1[j]==1 and np.random.rand() < G.edges[j,i]['lambda']*coeff_minus_1[j] and st[i]==0): 
                        st[i]=1
                        coeff_lam[i]=counter[i][0]
                        if flag_m : counter[i].pop(0)
        status_nodes.append(np.copy(st))
    return np.array(status_nodes)


def simulate_one_detSIR(G, s_type = "delta", S = 0.01, mask = ["SI"], T_max=100):
    """Function to simulate an epidemic using the deterministic-SIR model

    Args:
        G (nx.graph): Graph representing the contact network
        s_type (string): either "delta" or "n_sources", indicates how to interpret the parameter S
        S (int/float): number of infected/probability of being infected at time 0 (number/fraction of sources)
        mask (list): if it is equal to ["SI"], the function simulates an SI model, otherwise the i-th element of the
            list (between 0 and 1) represents the infectivity of the nodes at i timesteps after the infection
        T_max (int): maximum allowed time of simulation

    Returns:
        status_nodes (list): array of shape (T+1) x N containing the state of all the nodes from time 0 to time T
    """
    N=G.number_of_nodes()
    # Generate the sources
    status_nodes = []
    flag_s=0
    flag_m = 1
    flag_si = 0
    flag_sir = 0
    flag_dsir = 0
    if mask == ["SI"] : 
        mask = [1]
        flag_m = 0
        flag_si = 1
    elif len(mask) == T_max+1 : flag_sir = 1
    else: flag_dsir = 1
    counter = [mask.copy() for _ in range(N)]
    coeff_lam = np.ones(N)
    if s_type == "delta":
        s0 = []
        for i in range(N):
            if np.random.rand() < S : 
                s0.append(1)
                coeff_lam[i]=counter[i][0]
                if flag_m : counter[i].pop(0)
                flag_s =1
            else : s0.append(0)        
    else:
        flag_s=1
        s0=np.zeros(N)
        source_list = random.sample(range(N), S)
        s0[source_list]=1
        for i in source_list:
            coeff_lam[i]=counter[i][0]
            if flag_m : counter[i].pop(0)

    if (flag_s==0) : 
        s0[np.random.randint(0,N)]=1
        print("No sources... adding a single random source")
    status_nodes.append(np.array(s0))
    # Generate the epidemics
    st = np.copy(s0)
    while ((flag_dsir==1 and 1 in status_nodes[-1]) or ( flag_si==1 and 0 in status_nodes[-1]) or (flag_sir==1 and (0 in status_nodes[-1]) and (len(status_nodes) < T_max))) and (len(status_nodes) <= T_max):
        st_minus_1 = np.copy(st)
        coeff_minus_1 = coeff_lam.copy()
        for i in range(N):
            if st[i] == 1 :
                if not counter[i]: st[i]=2
                else :
                    coeff_lam[i]=counter[i][0]
                    if flag_m : counter[i].pop(0)
            elif st[i] == 0 :
                for j in nx.neighbors(G,i) :
                    if (st_minus_1[j]==1 and np.random.rand() < G.edges[j,i]['lambda']*coeff_minus_1[j] and st[i]==0): 
                        st[i]=1
                        coeff_lam[i]=counter[i][0]
                        if flag_m : counter[i].pop(0)
        status_nodes.append(np.copy(st))
    return np.array(status_nodes)

def simulate_one_SIR(G, s_type = "delta", S = 0.01, mu = 0, T_max=100, verbose=False):
    """Function to simulate an epidemic using conventional SIR model

    Args:
        G (nx.graph): Graph representing the contact network
        s_type (string): either "delta" or "n_sources", indicates how to interpret the parameter S
        S (int/float): number of infected/probability of being infected at time 0 (number/fraction of sources)
        mu (float): recovery parameter
        T_max (int): maximum allowed time of simulation
        verbose (bool): if True, print infection messages

    Returns:
        status_nodes (list): array of shape (T+1) x N containing the state of all the nodes from time 0 to time T
    """
    N=G.number_of_nodes()
    # Generate the sources
    status_nodes = []
    flag_s=0
    if s_type == "delta":
        s0 = []
        for i in range(N):
            if np.random.rand() < S : 
                s0.append(1)
                flag_s =1
                if verbose : print(f"node {i} is a source")
            else : s0.append(0)     

    else:
        flag_s=1
        s0=np.zeros(N)
        source_list = random.sample(range(N), S)
        if verbose : 
            for i in source_list:
                print(f"node {i} is a source")
        s0[source_list]=1

    if (flag_s==0) : 
        x = np.random.randint(0,N)
        s0[x]=1
        print("No sources... adding a single random source")
        if verbose : print(f"node {x} is a source")
    status_nodes.append(np.array(s0))
    # Generate the epidemics
    st = np.copy(s0)
    while ((mu>0. and 1 in status_nodes[-1]) or (mu==0. and sum(status_nodes[-1])!=N )) and (len(status_nodes) <= T_max):
        st_minus_1 = np.copy(st)
        for i in range(N):
            if st[i] == 1 :
                if random.random() < mu : 
                    st[i]=2
                    if verbose : print(f"node {i} recovered at time {len(status_nodes)-1}")
            elif st[i] == 0 :
                for j in nx.neighbors(G,i) :
                    if (st_minus_1[j]==1 and np.random.rand() < G.edges[j,i]['lambda'] and st[i]==0): 
                        st[i]=1
                        if verbose : print(f"node {i} infected by {j} at time {len(status_nodes)-1}")
        status_nodes.append(np.copy(st))
    return np.array(status_nodes)

def generate_contacts(G, T, lambda_, p_edge=1):
    """Function to generate contacts between nodes, given the probability lambda_

    Args:
        G (nx.graph): Graph representing the contact network
        T (int): time at which the epidemic stops
        lambda_ (float): probability of infection, constant for all edges
        p_edge (float): probability of having a contact


    Returns:
        contacts (list): List of directed contacts, each of the form (i,j,t,lambda)
    """
    contacts = []
    for t in range(T):
        for e in G.edges():
            if random.random() <= p_edge:
                contacts.append((e[0], e[1], t, lambda_))
                contacts.append((e[1], e[0], t, lambda_))
    return contacts

def generate_snapshot_obs_old(conf, o_type="rho", M=0.0, snap_time=-1):
    """Function to generate a snapshot observation, given an epidemic simulation

    Args:
        conf (array): array of shape (T+1) x N contaning the states of all the nodes from time 0 to time T
        o_type (string): either "rho" or "n_obs", indicates how to interpret the parameter M
            M (int/float): number of observed nodes/probability of being randomly observed 
        snap_time (int): time at which to take the snapshot. If not specified, this is a random int between 0 and T. 
    Returns:
        obs_sim (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
        fS (float): fraction of susceptible nodes when taking the tests
        fI (float): fraction of infected nodes when taking the tests
        tS (int): random time at which the tests were taken
    """
    obs_sim = []
    N = len(conf[0])
    T = len(conf) - 1
    if snap_time == -1: snap_time = random.choice(range(T+1))
    #elif snap_time > T: snap_time = T
    if snap_time < T:
        fS = np.count_nonzero(conf[snap_time] == 0)/N
        fI = np.count_nonzero(conf[snap_time] == 1)/N
        if o_type == "rho":
            obs_sim = [ (i, conf[snap_time,i], snap_time) for i in range(N) if np.random.random() < M]
        else: 
            obs_list = random.sample(range(N), M)
            obs_sim = [ (i, conf[snap_time,i], snap_time) for i in range(N) if i in obs_list]
    else:
        fS = np.count_nonzero(conf[T] == 0)/N
        fI = np.count_nonzero(conf[T] == 1)/N
        if o_type == "rho":
            obs_sim = [ (i, conf[T,i], snap_time) for i in range(N) if np.random.random() < M]
        else: 
            obs_list = random.sample(range(N), M)
            obs_sim = [ (i, conf[T,i], snap_time) for i in range(N) if i in obs_list]
    return obs_sim, fS, fI, snap_time

def generate_snapshot_obs(conf, o_type="rho", M=0.0, snap_time=-1, i_u_t=-1):
    """Function to generate a snapshot observation, given an epidemic simulation

    Args:
        conf (array): array of shape (T+1) x N contaning the states of all the nodes from time 0 to time T
        o_type (string): either "rho" or "n_obs", indicates how to interpret the parameter M
            M (int/float): number of observed nodes/probability of being randomly observed 
        snap_time (int): time at which to take the snapshot. If not specified, this is a random int between 0 and T. If the the time is not int, 
            we use the following formula: t1=flor(T), p=T-t1, and we extract with probability (1-p) observations at time t1 and with probability p at time t1+1. 
            On average we have observations at time T. 
        i_u_t (int): if -1, generate the snapshot at snap_time. If not, generate the snapshot at time t=min(snap_time,i_u_t)

    Returns:
        obs_sim (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
        fS (float): fraction of susceptible nodes when taking the tests
        fI (float): fraction of infected nodes when taking the tests
        t2 (int): Time at which the tests were taken
    """
    N = len(conf[0])
    T = len(conf) - 1
    if i_u_t != -1: snap_time = min(snap_time,i_u_t)
    if snap_time > T:
        print("warning snap_time > T, observation at T")

    obs_sim = []
    if snap_time == -1: snap_time = random.choice(range(T+1))
    #elif snap_time > T: snap_time = T
    t1 = math.floor(snap_time)
    t2 = math.ceil(snap_time)
    pp = snap_time-t1

    if snap_time <= T:
        fS1 = np.count_nonzero(conf[t1] == 0)/N
        fS2 = np.count_nonzero(conf[t2] == 0)/N
        fI1 = np.count_nonzero(conf[t1] == 1)/N
        fI2 = np.count_nonzero(conf[t2] == 1)/N
        fS = (1-pp) * fS1 + pp * fS2
        fI = (1-pp) * fI1 + pp * fI2
        if o_type == "rho":
            all_nodes = np.random.permutation(N)
            nodes1 = all_nodes[:int(N*(1-pp))]
            nodes2 = all_nodes[int(N*(1-pp)):]
            obs_sim = [ (i, conf[t1,i], t1) for i in nodes1 if np.random.random() < M]
            obs_sim.extend([ (i, conf[t2,i], t2) for i in nodes2 if np.random.random() < M])
        else: 
            obs_list = random.sample(range(N), M)
            nodes1 = obs_list[:int(M*(1-pp))]
            nodes2 = obs_list[int(M*(1-pp)):]
            obs_sim = [ (i, conf[t1,i], t1) for i in nodes1]
            obs_sim.extend([ (i, conf[t2,i], t2) for i in nodes2])
    else:
        fS = np.count_nonzero(conf[T] == 0)/N
        fI = np.count_nonzero(conf[T] == 1)/N
        if o_type == "rho":
            all_nodes = np.random.permutation(N)
            nodes1 = all_nodes[:int(N*(1-pp))]
            nodes2 = all_nodes[int(N*(1-pp)):]
            obs_sim = [ (i, conf[T,i], t1) for i in nodes1 if np.random.random() < M]
            obs_sim.extend([ (i, conf[T,i], t2) for i in nodes2 if np.random.random() < M])
        else: 
            obs_list = random.sample(range(N), M)
            nodes1 = obs_list[:int(M*(1-pp))]
            nodes2 = obs_list[int(M*(1-pp)):]
            obs_sim = [ (i, conf[T,i], t1) for i in nodes1]
            obs_sim.extend([ (i, conf[T,i], t2) for i in nodes2])
    return obs_sim, fS, fI, t2

#def obs_toSIB(obs):
#    """Function to convert a list of observations into a SIB array
#
#    Args:
#        obs (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
#        T (int): maximum time of infection
#        N (int): number of nodes
#
#    Returns:
#        SIB (array): array of shape (T+1) x N containing the observations
#    """
#    obs_temp = [(o[0],sib.Test(o[1]==0,o[1]==1,o[1]==1),o[2]) for o in obs]
#    return obs_temp
#
def generate_sensors_obs(conf, o_type="rho", M=0.0, T_max=100):
    """Function to generate a list of observations of a certain fraction of nodes, given an epidemic simulation

    Args:
        conf ([type]): [description]
        o_type (string): either "rho" or "n_obs", indicates how to interpret the parameter M
        M (int/float): number of observed nodes/probability of being randomly observed 
        T_max (int): maximum inferred time of infection by BP

    Returns:
        obs_sim (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
    """
    obs_sim = []
    conf = conf[:T_max+1]
    N = conf.shape[1]
    #T = conf.shape[0] - 1
    if o_type == "n_obs":
        obs_list = random.sample(range(N), M)
    for i in range(N):
        if ((o_type == "rho") and (np.random.random() < M)) or ((o_type == "n_obs") and (i in obs_list)):
            t_inf = np.nonzero(conf[:, i] == 1)[0]
            t_rec = np.nonzero(conf[:, i] == 2)[0]
            if len(t_inf) == 0:
                obs_temp = (i, 0, T_max)
                obs_sim.append(obs_temp)
            else:
                obs_temp = (i, 1, t_inf[0])
                obs_sim.append((obs_temp))
                if t_inf[0] > 0:
                    obs_temp = (i, 0, t_inf[0] - 1)
                    obs_sim.append((obs_temp))
                if ( (len(t_rec) == 0) & (t_inf[0]!=T_max) ): #This is superflous for SI, but does not change a thing
                    obs_temp = (i, 1, T_max)
                    obs_sim.append(obs_temp)
                if (len(t_rec) > 0):
                    obs_temp = (i, 1, t_rec[0] - 1)
                    obs_sim.append((obs_temp))
                    obs_temp = (i, 2, t_rec[0])
                    obs_sim.append((obs_temp))
    obs_sim = sorted(obs_sim, key=lambda tup: tup[2])
    return obs_sim