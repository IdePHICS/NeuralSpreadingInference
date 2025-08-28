import numpy as np
from Modules.st import *

epsilon =  2**(-400)
class FactorGraph:
    """Class to update the BP messages for the SI model"""

    def __init__(
        self, N, T, contacts, obs, delta, mask=["SI"], mask_type="SI", verbose=False
    ):
        """Construction of the FactorGraph object, starting from contacts and observations

        Args:
            N (int): Number of nodes in the contact network
            T (int): Time at which the simulation stops
            contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
            obs (list): List of the observations, each given by a list (i, 0/1, t), where 0 corresponds to S and 1 to I
            delta (float): Probability for an individual to be a source
            mask (list): if it is equal to ["SI"], the function simulates an SI model, otherwise the i-th element of the
                list (between 0 and 1) represents the infectivity of the nodes at i timesteps after the infection
            mask_type (string): Type of inference model. If equal to "SIR", it means we are simulating a SIR model
                and inferring using the dSIR model
        """
        # We create the messages tensor with uniform values
        self.messages = SparseTensor(N, T, contacts)
        if verbose:
            print("Messages matrices created")

        self.size = N
        self.time = T
        self.delta = delta
        self.contacts = contacts
        self.delta0 = delta

        # We create the Lambda matrices with one values
        self.Lambda0 = SparseTensor(
            Tensor_to_copy=self.messages
        ) 
        self.Lambda1 = SparseTensor(Tensor_to_copy=self.messages)
        if verbose:
            print("Lambdas matrices created")

        # We compute the Lambda matrices
        if mask == ["SI"]:
            compute_Lambdas(self.Lambda0, self.Lambda1, contacts)
        else:
            compute_Lambdas_dSIR(self.Lambda0, self.Lambda1, contacts, mask)
        if verbose:
            print("Lambdas matrices computed")

        # We create the observations array
        self.observations = np.ones(
            (self.size, self.time + 2)
        )
        # We reset the observations, starting from the obs list
        self.reset_obs(obs)
        if verbose:
            print("Observations array created")

        # We create some useful arrays
        self.out_msgs = np.array([], dtype="int") # indices of outgoing messages
        self.inc_msgs = np.array([], dtype="int") # indices of incoming messages
        self.repeat_deg = np.array([], dtype="int") # N-array of degrees
        self.obs_i = np.array([], dtype="int") # (Nxd)-array of i repeated d times 

        for i in range(len(self.messages.idx_list)): # loop through all nodes
            # add indices of messages incoming to node i to self.inc_msgs
            self.inc_msgs = np.concatenate(
                (self.inc_msgs, self.messages.idx_list[i]), axis=0
            )
            num_neighbours = len(self.messages.idx_list[i])
            # add number of neighbours to self.repeat_deg
            self.repeat_deg = np.append(self.repeat_deg, num_neighbours)
            for j in range(num_neighbours):
                self.obs_i = np.concatenate((self.obs_i, np.array([i]))) # add i repeated num_neighbours times to self.obs_i
                # add indices of messages outgoing from node i to self.out_msgs
                k = self.messages.adj_list[i][j] #index of the neighbour
                jnd = np.where(np.array(self.messages.adj_list[k]) == i)[0] # 1-array of the index of i in the neighbour's adjacency list
                self.out_msgs = np.concatenate(
                    (self.out_msgs, self.messages.idx_list[k][jnd]), axis=0
                )
        # array of indexes necessary in the function reduceat()
        # [0,d_1,d_1+d_2,...,d_1+...+d_{N-1}]
        self.reduce_idxs = np.delete(np.cumsum(self.repeat_deg), -1)
        self.reduce_idxs = np.concatenate(
            (np.array([0]), self.reduce_idxs), axis=0)

        if verbose:
            print("Lists of neighbors created")

        # define Lambda tensors used in update
        self.Lambda0_tilde = np.copy(self.Lambda0.values[self.inc_msgs])
        self.Lambda1_tilde = np.copy(self.Lambda1.values[self.inc_msgs])
        if verbose:
            print("Copied lambdas matrices computed")

    def get_gamma(self, arr, reduce_idxs, repeat_deg):
        """Function to compute ratios between gamma functions in iterate()

        Args:
            arr (array): initial array
            reduce_idxs (array): list of indexes necessary in the function reduceat()
            repeat_deg (array): N-array of degrees necessary in the function repeat()

        Returns:
            arr_5 (array): final array
        """
        epsilon = 1e-20
        # print(epsilon)
        arr[arr == 0] = epsilon
        arr_2 = np.log(arr)
        arr_copy = np.copy(arr_2)
        arr_3 = np.add.reduceat(arr_2, reduce_idxs, axis=0)
        arr_4 = np.repeat(arr_3, repeat_deg, axis=0)
        arr_5 = np.exp(arr_4 - arr_copy)
        return arr_5

    def iterate_old(self, damp):
        """Single iteration of the Belief Propagation algorithm

        Returns:
            err_max (float): Maximum difference between the messages at two consecutive iterations
            err_avg (float): Average difference between the messages at two consecutive iterations
        """
        T = self.time
        old_msgs = np.copy(self.messages.values)
        msgs_tilde = old_msgs[self.inc_msgs]

        # calculate gamma matrices
        gamma0_hat = np.sum(
            msgs_tilde * self.Lambda0_tilde, axis=1, keepdims=1)
        gamma1_hat = np.sum(
            msgs_tilde * self.Lambda1_tilde, axis=1, keepdims=1)
        gamma0 = self.get_gamma(gamma0_hat, self.reduce_idxs, self.repeat_deg)
        gamma1 = self.get_gamma(gamma1_hat, self.reduce_idxs, self.repeat_deg)

        # calculate part one of update
        one_obs = (1 - self.delta) * np.reshape(
            self.observations[self.obs_i], (len(self.out_msgs), 1, T + 2)
        )
        # due to floating point errors accrued when using np.log and np.exp
        # the substraction can sometimes give an extemely small negative result.
        # Therefore a hard cutoff is implemented to bring these values to zero.
        one_main = np.clip(
            self.Lambda1_tilde * gamma1 - self.Lambda0_tilde * gamma0, 0, 1
        )
        one = np.transpose(one_obs * one_main, (0, 2, 1))[:, 1: T + 1, :]

        # calculate part two of update
        two_obs = self.delta * self.observations[self.obs_i][:, 0]
        two_msgs = np.sum(msgs_tilde[:, :, 0], axis=1)
        two_main = self.get_gamma(two_msgs, self.reduce_idxs, self.repeat_deg)
        two = np.reshape(
            np.tile(np.reshape(two_obs * two_main,
                    (len(self.out_msgs), 1)), T + 2),
            (len(self.out_msgs), 1, T + 2),
        )

        # calculate part three of update
        three_obs = (1 - self.delta) * np.reshape(
            self.observations[self.obs_i][:, T + 1], (len(self.out_msgs), 1)
        )
        gamma1_reshaped = np.reshape(
            gamma1[:, 0, T + 1], (len(self.out_msgs), 1))
        three_main = self.Lambda1_tilde[:, :, T + 1] * gamma1_reshaped
        three = np.reshape(three_obs * three_main,
                           (len(self.out_msgs), 1, T + 2))

        # update the message values
        update_one = np.concatenate(
            (
                np.zeros((len(self.out_msgs), 1, T + 2)),
                one,
                np.zeros((len(self.out_msgs), 1, T + 2)),
            ),
            axis=1,
        )
        update_two = np.concatenate(
            (two, np.zeros((len(self.out_msgs), T + 1, T + 2))), axis=1
        )
        update_three = np.concatenate(
            (np.zeros((len(self.out_msgs), T + 1, T + 2)), three), axis=1
        )
        new_msgs = update_one + update_two + update_three
        norm = np.reshape(np.sum(new_msgs, axis=(1, 2)),
                          (len(self.out_msgs), 1, 1))
        norm_msgs = new_msgs / norm
        self.messages.values[self.out_msgs] = (1 - damp) * norm_msgs + damp * old_msgs[
            self.out_msgs
        ]  # Add dumping
        err_array = np.abs(old_msgs - self.messages.values)

        return err_array.max(), err_array.mean()

    def iterate_plus_old(self, damp):
        """Single iteration of the Belief Propagation algorithm

        Returns:
            err_max (float): Maximum difference between the messages at two consecutive iterations
            err_avg (float): Average difference between the messages at two consecutive iterations
        """
        T = self.time
        old_msgs = np.copy(self.messages.values)
        msgs_tilde = old_msgs[self.inc_msgs]

        # calculate gamma matrices
        gamma0_hat = np.sum(
            msgs_tilde * self.Lambda0_tilde, axis=1, keepdims=1)
        gamma1_hat = np.sum(
            msgs_tilde * self.Lambda1_tilde, axis=1, keepdims=1)
        gamma0 = self.get_gamma(gamma0_hat, self.reduce_idxs, self.repeat_deg)
        gamma1 = self.get_gamma(gamma1_hat, self.reduce_idxs, self.repeat_deg)

        # part for BP-AMP
        gamma0p = np.multiply.reduceat(np.sum(
            msgs_tilde * self.Lambda0_tilde, axis=1), self.reduce_idxs, axis=0)
        gamma1p = np.multiply.reduceat(np.sum(
            msgs_tilde * self.Lambda1_tilde, axis=1), self.reduce_idxs, axis=0)
        gamma0p[:,-1] = np.zeros_like(gamma0p[:,-1])
        gammap = gamma1p-gamma0p
        nu_plus = np.sum(gammap[:,1:]*self.observations[:,1:],axis=1)
        nu_minus = self.observations[:,0]*np.multiply.reduceat(np.sum(self.messages.values[:,:,0],axis=1), self.reduce_idxs, axis=0)
        #nu = nu_plus/(nu_plus + nu_minus)
        nu = np.array([1. if (nu_minus[i] < 1e-25) else (nu_plus[i]/(nu_plus[i] + nu_minus[i])) for i in range(self.size)])
    
        # calculate part one of update
        one_obs = np.reshape((np.multiply(1 - self.delta,self.observations.T).T)[self.obs_i], (len(self.out_msgs), 1, T + 2)
        )
        # due to floating point errors accrued when using np.log and np.exp
        # the substraction can sometimes give an extemely small negative result.
        # Therefore a hard cutoff is implemented to bring these values to zero.
        one_main = np.clip(
            self.Lambda1_tilde * gamma1 - self.Lambda0_tilde * gamma0, 0, 1
        )
        one = np.transpose(one_obs * one_main, (0, 2, 1))[:, 1: T + 1, :]

        # calculate part two of update
        two_obs = (np.multiply(self.delta,self.observations.T).T)[self.obs_i][:, 0]
        two_msgs = np.sum(msgs_tilde[:, :, 0], axis=1)
        two_main = self.get_gamma(two_msgs, self.reduce_idxs, self.repeat_deg)
        two = np.reshape(
            np.tile(np.reshape(two_obs * two_main,
                    (len(self.out_msgs), 1)), T + 2),
            (len(self.out_msgs), 1, T + 2),
        )

        # calculate part three of update
        three_obs =  np.reshape(
            (np.multiply(1 - self.delta,self.observations.T).T)[self.obs_i][:, T + 1], (len(self.out_msgs), 1)
        )
        gamma1_reshaped = np.reshape(
            gamma1[:, 0, T + 1], (len(self.out_msgs), 1))
        three_main = self.Lambda1_tilde[:, :, T + 1] * gamma1_reshaped
        three = np.reshape(three_obs * three_main,
                           (len(self.out_msgs), 1, T + 2))

        # update the message values
        update_one = np.concatenate(
            (
                np.zeros((len(self.out_msgs), 1, T + 2)),
                one,
                np.zeros((len(self.out_msgs), 1, T + 2)),
            ),
            axis=1,
        )
        update_two = np.concatenate(
            (two, np.zeros((len(self.out_msgs), T + 1, T + 2))), axis=1
        )
        update_three = np.concatenate(
            (np.zeros((len(self.out_msgs), T + 1, T + 2)), three), axis=1
        )
        new_msgs = update_one + update_two + update_three
        norm = np.reshape(np.sum(new_msgs, axis=(1, 2)),
                          (len(self.out_msgs), 1, 1))
        if (np.sum(norm==0)!=0): print(f"zeros=={np.sum(norm==0)}")
        norm_msgs = new_msgs / norm
        self.messages.values[self.out_msgs] = (1 - damp) * norm_msgs + damp * old_msgs[
            self.out_msgs
        ]  # Add dumping
        err_array = np.abs(old_msgs - self.messages.values)

        return err_array.max(), err_array.mean(), nu

    def iterate_plus(self, damp):
        """Single iteration of the Belief Propagation algorithm

        Returns:
            err_max (float): Maximum difference between the messages at two consecutive iterations
            err_avg (float): Average difference between the messages at two consecutive iterations
        """
        T = self.time
        old_msgs = np.copy(self.messages.values)
        msgs_tilde = old_msgs[self.inc_msgs]

        # calculate gamma matrices
        gamma0_hat = np.sum(
            msgs_tilde * self.Lambda0_tilde, axis=1, keepdims=1)
        gamma1_hat = np.sum(
            msgs_tilde * self.Lambda1_tilde, axis=1, keepdims=1)
        gamma0 = self.get_gamma(gamma0_hat, self.reduce_idxs, self.repeat_deg)
        gamma1 = self.get_gamma(gamma1_hat, self.reduce_idxs, self.repeat_deg)

        # part for BP-AMP
        gamma0p = np.multiply.reduceat(np.sum(
            msgs_tilde * self.Lambda0_tilde, axis=1), self.reduce_idxs, axis=0)
        gamma1p = np.multiply.reduceat(np.sum(
            msgs_tilde * self.Lambda1_tilde, axis=1), self.reduce_idxs, axis=0)
        gamma0p[:,-1] = np.zeros_like(gamma0p[:,-1])
        gammap = gamma1p-gamma0p
        nu_plus = np.sum(gammap[:,1:]*self.observations[:,1:],axis=1)
        nu_minus = self.observations[:,0]*np.multiply.reduceat(np.sum(self.messages.values[:,:,0],axis=1), self.reduce_idxs, axis=0)
        #nu = nu_plus/(nu_plus + nu_minus)
        nu = np.array([1. if (nu_minus[i] < 1e-25) else (nu_plus[i]/(nu_plus[i] + nu_minus[i])) for i in range(self.size)])
    
        # calculate part one of update
        one_obs = np.reshape(
            (np.repeat((1 - self.delta), self.repeat_deg, axis=0) * self.observations[self.obs_i].T).T, 
            (len(self.out_msgs), 1, T + 2)
        )
        # due to floating point errors accrued when using np.log and np.exp
        # the substraction can sometimes give an extemely small negative result.
        # Therefore a hard cutoff is implemented to bring these values to zero.
        one_main = np.clip(
            self.Lambda1_tilde * gamma1 - self.Lambda0_tilde * gamma0, 0, 1
        )
        one = np.transpose(one_obs * one_main, (0, 2, 1))[:, 1: T + 1, :]

        # calculate part two of update
        two_obs = ((np.repeat(self.delta, self.repeat_deg, axis=0) * self.observations[self.obs_i].T).T)[:, 0]
        two_msgs = np.sum(msgs_tilde[:, :, 0], axis=1)
        two_main = self.get_gamma(two_msgs, self.reduce_idxs, self.repeat_deg)
        two = np.reshape(
            np.tile(np.reshape(two_obs * two_main,
                    (len(self.out_msgs), 1)), T + 2),
            (len(self.out_msgs), 1, T + 2),
        )

        # calculate part three of update
        three_obs = np.reshape(
            ((np.repeat((1 - self.delta), self.repeat_deg, axis=0) * self.observations[self.obs_i].T).T)[:, T + 1], 
            (len(self.out_msgs), 1)
        )
        gamma1_reshaped = np.reshape(
            gamma1[:, 0, T + 1], (len(self.out_msgs), 1))
        three_main = self.Lambda1_tilde[:, :, T + 1] * gamma1_reshaped
        three = np.reshape(three_obs * three_main,
                           (len(self.out_msgs), 1, T + 2))

        # update the message values
        update_one = np.concatenate(
            (
                np.zeros((len(self.out_msgs), 1, T + 2)),
                one,
                np.zeros((len(self.out_msgs), 1, T + 2)),
            ),
            axis=1,
        )
        update_two = np.concatenate(
            (two, np.zeros((len(self.out_msgs), T + 1, T + 2))), axis=1
        )
        update_three = np.concatenate(
            (np.zeros((len(self.out_msgs), T + 1, T + 2)), three), axis=1
        )
        new_msgs = update_one + update_two + update_three
        norm = np.reshape(np.sum(new_msgs, axis=(1, 2)),
                          (len(self.out_msgs), 1, 1))
        if (np.sum(norm==0)!=0): print(f"zeros=={np.sum(norm==0)}")
        norm_msgs = new_msgs / norm
        self.messages.values[self.out_msgs] = (1 - damp) * norm_msgs + damp * old_msgs[
            self.out_msgs
        ]  # Add dumping
        err_array = np.abs(old_msgs - self.messages.values)

        return err_array.max(), err_array.mean(), nu

    def compute_nu(self):
        """Function to compute the nu values in the BP-AMP algorithm

        Returns:
            nu (array): array of values of nu(+1)
        """
        old_msgs = np.copy(self.messages.values)
        msgs_tilde = old_msgs[self.inc_msgs]
        #print(min(msgs_tilde.flatten()),max(msgs_tilde.flatten()),np.isnan(msgs_tilde).sum())
        # part for BP-AMP
        gamma0p = np.multiply.reduceat(np.sum(
            msgs_tilde * self.Lambda0_tilde, axis=1), self.reduce_idxs, axis=0)
        gamma1p = np.multiply.reduceat(np.sum(
            msgs_tilde * self.Lambda1_tilde, axis=1), self.reduce_idxs, axis=0)
        gamma0p[:,-1] = np.zeros_like(gamma0p[:,-1])
        gammap = gamma1p-gamma0p
        #print(min(gamma0p.flatten()),max(gamma0p.flatten()),np.isnan(gamma0p).sum())
        #print(min(gamma1p.flatten()),max(gamma1p.flatten()),np.isnan(gamma1p).sum())
        nu_plus = np.sum(gammap[:,1:]*self.observations[:,1:],axis=1)
        nu_minus = self.observations[:,0]*np.multiply.reduceat(np.sum(self.messages.values[:,:,0],axis=1), self.reduce_idxs, axis=0)
        #nu = nu_plus/(nu_plus + nu_minus)
        #print(min(norm),max(norm),np.isnan(norm).sum())
        norm = np.maximum(nu_plus + nu_minus,epsilon)
        return nu_plus, norm
    
    def iterate(self, damp):
        """Single iteration of the Belief Propagation algorithm, without computing nu

        Returns:
            err_max (float): Maximum difference between the messages at two consecutive iterations
            err_avg (float): Average difference between the messages at two consecutive iterations
        """
        T = self.time
        old_msgs = np.copy(self.messages.values)
        msgs_tilde = old_msgs[self.inc_msgs]

        # calculate gamma matrices
        gamma0_hat = np.sum(
            msgs_tilde * self.Lambda0_tilde, axis=1, keepdims=1)
        gamma1_hat = np.sum(
            msgs_tilde * self.Lambda1_tilde, axis=1, keepdims=1)
        gamma0 = self.get_gamma(gamma0_hat, self.reduce_idxs, self.repeat_deg)
        gamma1 = self.get_gamma(gamma1_hat, self.reduce_idxs, self.repeat_deg)
    
        # calculate part one of update
        one_obs = np.reshape(
            (np.repeat((1 - self.delta), self.repeat_deg, axis=0) * self.observations[self.obs_i].T).T, 
            (len(self.out_msgs), 1, T + 2)
        )
        # due to floating point errors accrued when using np.log and np.exp
        # the substraction can sometimes give an extemely small negative result.
        # Therefore a hard cutoff is implemented to bring these values to zero.
        one_main = np.clip(
            self.Lambda1_tilde * gamma1 - self.Lambda0_tilde * gamma0, 0, 1
        )
        one = np.transpose(one_obs * one_main, (0, 2, 1))[:, 1: T + 1, :]

        # calculate part two of update
        two_obs = ((np.repeat(self.delta, self.repeat_deg, axis=0) * self.observations[self.obs_i].T).T)[:, 0]
        two_msgs = np.sum(msgs_tilde[:, :, 0], axis=1)
        two_main = self.get_gamma(two_msgs, self.reduce_idxs, self.repeat_deg)
        two = np.reshape(
            np.tile(np.reshape(two_obs * two_main,
                    (len(self.out_msgs), 1)), T + 2),
            (len(self.out_msgs), 1, T + 2),
        )

        # calculate part three of update
        three_obs = np.reshape(
            ((np.repeat((1 - self.delta), self.repeat_deg, axis=0) * self.observations[self.obs_i].T).T)[:, T + 1], 
            (len(self.out_msgs), 1)
        )
        gamma1_reshaped = np.reshape(
            gamma1[:, 0, T + 1], (len(self.out_msgs), 1))
        three_main = self.Lambda1_tilde[:, :, T + 1] * gamma1_reshaped
        three = np.reshape(three_obs * three_main,
                           (len(self.out_msgs), 1, T + 2))

        # update the message values
        update_one = np.concatenate(
            (
                np.zeros((len(self.out_msgs), 1, T + 2)),
                one,
                np.zeros((len(self.out_msgs), 1, T + 2)),
            ),
            axis=1,
        )
        update_two = np.concatenate(
            (two, np.zeros((len(self.out_msgs), T + 1, T + 2))), axis=1
        )
        update_three = np.concatenate(
            (np.zeros((len(self.out_msgs), T + 1, T + 2)), three), axis=1
        )
        new_msgs = update_one + update_two + update_three
        norm = np.reshape(np.sum(new_msgs, axis=(1, 2)),
                          (len(self.out_msgs), 1, 1))
        #norm = np.maximum(epsilon,norm)
        if (np.sum(norm==0)!=0): print(f"zeros=={np.sum(norm==0)}")
        #norm_msgs = np.minimum(new_msgs/norm,1-epsilon)
        #self.messages.values[self.out_msgs] = (1 - damp) * new_msgs/norm + damp * old_msgs[
        #    self.out_msgs
        #]  # Add dumping
        self.messages.values[self.out_msgs] = np.array([  (1-damp)*new_msgs[i]/norm[i] + damp*old_msgs[idx] if norm[i] > 0 else old_msgs[idx] for i,idx in enumerate(self.out_msgs)])
        err_array = np.abs(old_msgs - self.messages.values)

        return err_array.max(), err_array.mean()

    def update(self, maxit=100, tol=1e-6, damp=0.0, print_iter=None):
        """Multiple iterations of the BP algorithm through the method iterate()

        Args:
            maxit (int, optional): Maximum number of iterations of BP. Defaults to 100.
            tol (double, optional): Tolerance threshold for the difference between consecutive. Defaults to 1e-6.

        Returns:
            i (int): Iteration at which the algorithm stops
            error (float): Error on the messages at the end of the iterations
        """
        i = 0
        error_mean = 1
        while i < maxit and error_mean > (1-damp)*tol:
            error_max, error_mean = self.iterate(damp)
            i += 1
            if print_iter != None:
                print_iter([error_max, error_mean], i)
        return i, [error_max, error_mean]

    def marginals(self):
        """Computes the array of the BP marginals for each node

        Returns:
            marginals (np.array): Array of the BP marginals, of shape N x (T+2)
        """

        marginals = []
        # we have one marginal (Tx1 vector) for each node.
        for n in range(self.size):  # loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            inc_msg = self.messages.values[
                inc_indices[0]
            ]  # b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j.
            out_msg = self.messages.values[out_indices[0]]
            marg = np.sum(
                inc_msg * np.transpose(out_msg), axis=0
            )  # transpose outgoing message so index to sum over after broadcasting is 0.
            # DEBUG CHECK
            # if (marg.sum()==0):
            #    print(f"INC{inc_msg}")
            #    print(f"OUT{np.transpose(out_msg)}")
            sum = max(marg.sum(),epsilon)
            marginals.append(marg / sum )
        return np.asarray(marginals)

    def pair_marginals(self):
        """Computes the array of the BP messages for each node

        Returns:
            messages (np.array): Array of the BP messages, of shape E x (T+2) x (T+2)
        """
        pair_marg = SparseTensor(
            Tensor_to_copy=self.messages
        )
        pair_marg.values = self.messages.values[self.out_msgs] * (
            np.transpose(self.messages.values, axes=(0, 2, 1))[self.inc_msgs])
        pair_marg.values = pair_marg.values / np.sum(pair_marg.values, axis=(1, 2))[:, np.newaxis, np.newaxis]
        return pair_marg

    def get_messages(self):
        """Returns a copy of the BP messages

        Returns:
            marginals (np.array): Array of the BP marginals, of shape N x (T+2)
        """
        return np.copy(self.messages.values)
    
    def loglikelihood_BP(self):
        """Computes the LogLikelihood from the BP messages

        Returns:
            logL (float): LogLikelihood
        """
        T = self.time
        N = self.size

        log_zi = 0.0
        #dummy_array = np.zeros((1, T + 2))
        _, zi_tab = self.compute_nu()
        for i in range(self.size):
            log_zi = log_zi + np.log(max(zi_tab[i],epsilon))

        log_zij = 0.0
        # we have one marginal (Tx1 vector) for each node.
        for n in range(self.size):  # loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            for j in np.arange(0, len(out_indices)):
                inc_msg = self.messages.values[
                    inc_indices[j]
                ]  # b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j.
                out_msg = self.messages.values[out_indices[j]]
                marg = np.sum(
                    inc_msg * np.transpose(out_msg), axis=0
                )  # transpose outgoing message so index to sum over after broadcasting is 0.
                log_zij = log_zij + np.log(marg.sum())
        return (log_zi - 0.5 * log_zij )/N
    
    def loglikelihood(self):
        """Computes the LogLikelihood from the BP messages

        Returns:
            logL (float): LogLikelihood
        """
        T = self.time
        N = self.size

        log_zi = 0.0
        dummy_array = np.zeros((1, T + 2))
        for i in range(self.size):
            inc_indices, out_indices = self.messages.get_all_indices(i)
            inc_msgs = self.messages.get_neigh_i(i)
            inc_lambda0 = self.Lambda0.get_neigh_i(i)
            inc_lambda1 = self.Lambda1.get_neigh_i(i)

            gamma0_ki = np.reshape(
                np.prod(np.sum(inc_lambda0 * inc_msgs, axis=1),
                        axis=0), (1, T + 2)
            )
            gamma1_ki = np.reshape(
                np.prod(np.sum(inc_lambda1 * inc_msgs, axis=1),
                        axis=0), (1, T + 2)
            )
            dummy_array = np.transpose(
                (
                    (1 - self.delta[i])
                    * np.reshape(self.observations[i], (1, T + 2))
                    * (gamma1_ki - gamma0_ki)
                )
            )
            dummy_array[0] = (
                self.delta[i]
                * self.observations[i][0]
                * np.prod(np.sum(inc_msgs[:, :, 0], axis=1), axis=0)
            )
            dummy_array[T + 1] = np.transpose(
                (1 - self.delta[i]) *
                self.observations[i][T + 1] * gamma1_ki[0][T + 1]
            )
            log_zi = log_zi + np.log(dummy_array.sum())

        log_zij = 0.0
        # we have one marginal (Tx1 vector) for each node.
        for n in range(self.size):  # loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            for j in np.arange(0, len(out_indices)):
                inc_msg = self.messages.values[
                    inc_indices[j]
                ]  # b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j.
                out_msg = self.messages.values[out_indices[j]]
                marg = np.sum(
                    inc_msg * np.transpose(out_msg), axis=0
                )  # transpose outgoing message so index to sum over after broadcasting is 0.
                log_zij = log_zij + np.log(marg.sum())
        return (log_zi - 0.5 * log_zij )/N
    

    #def loglikelihood(self): # not implemented
    #    """Computes the LogLikelihood from the BP messages

   #     Returns:
    #        logL (float): LogLikelihood
    #    """

    #    return 0.0
        #T = self.time
#
        #log_zi = 0.0
        #dummy_array = np.zeros((1, T + 2))
        #for i in range(self.size):
        #    inc_indices, out_indices = self.messages.get_all_indices(i)
        #    inc_msgs = self.messages.get_neigh_i(i)
        #    inc_lambda0 = self.Lambda0.get_neigh_i(i)
        #    inc_lambda1 = self.Lambda1.get_neigh_i(i)
#
        #    gamma0_ki = np.reshape(
        #        np.prod(np.sum(inc_lambda0 * inc_msgs, axis=1),
        #                axis=0), (1, T + 2)
        #    )
        #    gamma1_ki = np.reshape(
        #        np.prod(np.sum(inc_lambda1 * inc_msgs, axis=1),
        #                axis=0), (1, T + 2)
        #    )
        #    dummy_array = np.transpose(
        #        (
        #            (1 - self.delta[i])
        #            * np.reshape(self.observations[i], (1, T + 2))
        #            * (gamma1_ki - gamma0_ki)
        #        )
        #    )
        #    dummy_array[0] = (
        #        self.delta[i]
        #        * self.observations[i][0]
        #        * np.prod(np.sum(inc_msgs[:, :, 0], axis=1), axis=0)
        #    )
        #    dummy_array[T + 1] = np.transpose(
        #        (1 - self.delta[i]) *
        #        self.observations[i][T + 1] * gamma1_ki[0][T + 1]
        #    )
        #    log_zi = log_zi + np.log(dummy_array.sum())
#
        #log_zij = 0.0
        ## we have one marginal (Tx1 vector) for each node.
        #for n in range(self.size):  # loop through all nodes
        #    inc_indices, out_indices = self.messages.get_all_indices(n)
        #    for j in np.arange(0, len(out_indices)):
        #        inc_msg = self.messages.values[
        #            inc_indices[j]
        #        ]  # b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j.
        #        out_msg = self.messages.values[out_indices[j]]
        #        marg = np.sum(
        #            inc_msg * np.transpose(out_msg), axis=0
        #        )  # transpose outgoing message so index to sum over after broadcasting is 0.
        #        log_zij = log_zij + np.log(marg.sum())
        #return log_zi - 0.5 * log_zij

    def reset_obs(self, obs):
        """Resets the observations, starting from the obs list

        Args:
            obs (list): List of the observations, each given by a list (i, 0/1, t), where 0 corresponds to S and 1 to I

        """
        self.observations = np.ones(
            (self.size, self.time + 2)
        )  
        # creating the mask for observations
        for o in obs:
            if o[1] == 0:
                self.observations[o[0]][: o[2] + 1] = 0
            else:
                self.observations[o[0]][o[2] + 1:] = 0

    def set_delta(self, delta):
        """Set a new value for delta

        Args:
            delta (float): value for delta

        """
        self.delta = delta

    def get_delta0(self):
        """Get the value of delta

        Returns:
            delta (float): value of delta

        """
        return self.delta0
    
    def get_delta(self):
        """Get the value of delta

        Returns:
            delta (float): value of delta

        """
        return self.delta


class pop_dyn(FactorGraph):
    def pop_dyn_RRG(self, c=3):
        """Single iteration of the Population dynamics algorithm for a d-RRG

        Args:
            c (int): degree of the RRG

        Returns:
            difference (float): Maximum difference between the messages at two consecutive iterations
        """
        T = self.time
        N = self.size
        old_msgs = np.copy(self.messages.values)
        for i in range(N):
            indices = [np.random.randint(0, N) for _ in range(c - 1)]
            inc_msgs = np.array([old_msgs[idx] for idx in indices])
            inc_lambda0 = np.array([self.Lambda0.values[idx]
                                   for idx in indices])
            inc_lambda1 = np.array([self.Lambda1.values[idx]
                                   for idx in indices])
            gamma0_ki = np.reshape(
                np.prod(np.sum(inc_lambda0 * inc_msgs, axis=1),
                        axis=0), (1, T + 2)
            )
            gamma1_ki = np.reshape(
                np.prod(np.sum(inc_lambda1 * inc_msgs, axis=1),
                        axis=0), (1, T + 2)
            )
            self.messages.values[i] = np.transpose(
                (
                    (1 - self.delta)
                    * np.reshape(np.ones(T + 2), (1, T + 2))
                    * (inc_lambda1[0] * gamma1_ki - inc_lambda0[0] * gamma0_ki)
                )
            )
            self.messages.values[i][0] = self.delta * np.prod(
                np.sum(inc_msgs[:, :, 0], axis=1), axis=0
            )
            self.messages.values[i][T + 1] = np.transpose(
                (1 - self.delta) *
                inc_lambda1[0][:, T + 1] * gamma1_ki[0][T + 1]
            )
            norm = self.messages.values[i].sum()  # normalize the messages
            self.messages.values[i] = self.messages.values[i] / norm

        difference = np.abs(old_msgs - self.messages.values).max()

        return difference
