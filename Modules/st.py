import numpy as np


class SparseTensor:
    """Class to represent an N x N x T x T sparse tensor as a 2 x num_edges x T x T full tensor"""

    def __init__(self, N=0, T=0, contacts=[], Tensor_to_copy=None):
        """Construction of the tensor. If no tensor is given, then calls init(), otherwise calls init_like()

        Args:
            N (int): Number of nodes in the contact network
            T (int): Value of the last simulation time
            contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
            Tensor_to_copy (SparseTensor): SparseTensor to copy to create a new object
        """
        if Tensor_to_copy is None:
            self.init(N, T, contacts)
        else:
            self.init_like(Tensor_to_copy)

    def init(self, N, T, contacts):
        """Initialization of the tensor, given the contacts

        Args:
            N (int): Number of nodes in the contact network
            T (int): Value of the last simulation time
            contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
        """
        # We initialize the idx_list and adj_list to empty lists
        self.idx_list = []
        self.adj_list = [[] for _ in range(N)]

        # We store the number of nodes and the last simulation time
        self.N = N
        self.T = T

        # We get the array of edges from the list of contacts
        if not contacts:
            edge_list = []
        else:       
            edge_list = np.unique(
                np.asarray(contacts, dtype="int")[:, :2], axis=0
            )  
            edge_list = np.concatenate(
                (edge_list, np.flip(edge_list, 1)), axis=0)
            edge_list = np.unique(edge_list, axis=0)           
        self.num_direct_edges = len(edge_list)

        # We populate the adjacency list
        for e in edge_list:
            self.adj_list[e[0]].append(e[1])
        self.degree = np.array([len(a) for a in self.adj_list])

        # We populate the N x d idx_list, containing the indices of the tensor
        c = 0
        for d in self.degree:
            self.idx_list.append(np.arange(c, c + d))
            c = c + d

        # We initialize the tensor with uniform values
        self.values = np.full(
            (self.num_direct_edges, T + 2, T + 2), 1 / ((T + 2) * (T + 2))
        )

    def init_like(self, Tensor):
        """Initialization of the tensor, given another tensor, putting all values to one

        Args:
            Tensor (SparseTensor): The Sparse Tensor object we want to copy
        """
        self.idx_list = Tensor.idx_list
        self.adj_list = Tensor.adj_list
        self.N = Tensor.N
        self.T = Tensor.T
        self.num_direct_edges = Tensor.num_direct_edges
        self.degree = Tensor.degree
        # We initialize the tensor with all values to one
        self.values = np.full(
            (self.num_direct_edges, self.T + 2, self.T + 2), 1.0)

    def get_idx_ij(self, i, j):
        """Returns index corresponding to the (i, j) entrance of the tensor

        Args:
            i (int): Index of the sending node
            j (int): Index of the receiving node

        Returns:
            idx (int): Index corresponding to the (i, j) entrance of the tensor
        """
        idx_i = self.adj_list[j].index(i)
        idx = self.idx_list[j][idx_i]

        return idx

    def get_ij(self, i, j):
        """Returns the T x T matrix corresponding to the (i, j) entrance of the tensor

        Args:
            i (int): Index of the sending node
            j (int): Index of the receiving node

        Returns:
            val_idx (float): Element of the array values corresponding to the (i, j) entrance of the tensor
        """
        idx_i = self.adj_list[j].index(i)
        idx = self.idx_list[j][idx_i]

        return self.values[idx]

    def get_neigh_i(self, i):
        """Returns d_i T x T matrices corresponding to the (*, i) entrances of the tensor

        Args:
            i (int): Index of the receiving node

        Returns:
            val_neigh (list): List of elements of the array values corresponding to the (*, i) entrances of the tensor
        """
        return self.values[self.idx_list[i]]

    def get_all_indices(self, i):
        """Returns all the indices corresponding to the incoming and outgoing messages for a given node 

        Args:
            i (int): Index of the node

        Returns:
            incoming_indices (list): List of indices corresponding to the (*, i) entrances of the tensor
            outgoing_indices (list): List of indices corresponding to the (i, *) entrances of the tensor
        """
        incoming_indices = self.idx_list[i]
        outgoing_indices = []
        for j in self.adj_list[i]:
            idx_i = self.adj_list[j].index(i)
            idx = self.idx_list[j][idx_i]
            outgoing_indices.append(idx)

        return incoming_indices, outgoing_indices


def compute_Lambdas(Lambda0, Lambda1, contacts):  # change to loop over full contacts
    """Computes (once and for all) the entrances of the tensors Lambda0 and Lambda1, starting from the list of contacts
    Args:
        Lambda0 (SparseTensor): SparseTensor useful to update the BP messages
        Lambda1 (SparseTensor): SparseTensor useful to update the BP messages
        contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
    """
    Lambda1.values[:] = 0
    Lambda0.values[:] = 0

    # 1) populate the tensor with the lambdas values, equal in each row
    for cc in contacts:
        # In Lambda1 the first two rows are zero
        Lambda1.get_ij(cc[0], cc[1])[:, cc[2] + 2] = cc[3]
        # In Lambda0 the first and last rows are zero
        Lambda0.get_ij(cc[0], cc[1])[:, cc[2] + 1] = cc[3]

    # fill the lower triangular matrix lambdas with 0, with different offsets
    n_dim = Lambda1.values.shape[1]
    a, b = np.tril_indices(n_dim, k=1) 
    Lambda1.values[:, a, b] = 0
    a, b = np.tril_indices(n_dim, k=0)
    Lambda0.values[:, a, b] = 0

    # takes 1 - lambdas
    Lambda1.values = 1 - Lambda1.values
    Lambda0.values = 1 - Lambda0.values

    # Makes the cumulative products over the rows to compute the final version of Lambdas matrices
    Lambda1.values = np.cumprod(Lambda1.values, axis=2)
    Lambda0.values = np.cumprod(Lambda0.values, axis=2)


# added mask to work with deterministic SIR model
def compute_Lambdas_dSIR(Lambda0, Lambda1, contacts, mask):
    """Computes (once and for all) the entrances of the tensors Lambda0 and Lambda1, starting from the list of contacts
    Args:
        Lambda0 (SparseTensor): SparseTensor useful to update the BP messages
        Lambda1 (SparseTensor): SparseTensor useful to update the BP messages
        contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
        mask (list): List of infectivity coefficients, given as [c_{t_j+1}, c_{t_j+2}, ...]
    """
    Tp2 = len(Lambda0.values[0][0])
    Lambda1.values[:] = 0
    Lambda0.values[:] = 0

    # populated the tensor with the lambdas values
    for cc in contacts:
        # print(cc[0], cc[1])
        Lambda1.get_ij(cc[0], cc[1])[:, cc[2] + 2] = cc[3]
        Lambda0.get_ij(cc[0], cc[1])[:, cc[2] + 1] = cc[3]

    # fill the lower triangular matrix lambdas with 0
    n_dim = Lambda1.values.shape[1]
    a, b = np.tril_indices(n_dim, k=1)
    Lambda1.values[:, a, b] = 0
    a, b = np.tril_indices(n_dim, k=0)
    Lambda0.values[:, a, b] = 0

    # Compute and apply the infectivity masks
    Mask1 = np.array([np.asarray(
        ([1]*(tj+2) + mask + [0]*(Tp2-len(mask)-tj-2))[:Tp2]) for tj in range(Tp2)])
    Mask0 = np.array([np.asarray(
        ([1]*(tj+1) + mask + [0]*(Tp2-len(mask)-tj-1))[:Tp2]) for tj in range(Tp2)])

    Lambda1.values[:] = Lambda1.values[:]*Mask1
    Lambda0.values[:] = Lambda0.values[:]*Mask0

    # takes 1 - lambdas
    Lambda1.values = 1 - Lambda1.values
    Lambda0.values = 1 - Lambda0.values

    # Makes the cumulative products to compute the final version of Lambdas matrices
    Lambda1.values = np.cumprod(Lambda1.values, axis=2)
    Lambda0.values = np.cumprod(Lambda0.values, axis=2)
