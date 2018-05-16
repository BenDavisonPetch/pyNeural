# pyNeural (Very WIP)

A collection of python 3 scripts built using only internal libraries that serves as a basic neural network.

*Derivation of backpropagation calculus (Also WIP)*

C = our cost function
a[L][i] = activation of node in layer L at index i
w[L][j][k] = weight between node[L][j] and node[L-1][k]

dC/da for all nodes in the output layer = 2(a-y) where y is the intended output for that node

Otherwise,
dC/da[L][i] = sum(dz[L+1][j]/da[L][i] * da[L+1][j]/dz[L+1][j] * dC/a[L+1][j]) for j = 0 -> (number of nodes in layer L+1)-1
            = sum(w[L+1][j][i] * a[L+1][j]*(1-a[L+1][j]) * dC/a[L+1][j]) for j ...

For biases:
dC/db[L][i] = dz[L][i]/db[L][i] * da[L][i]/dz[L][i] * dC/da[L][i]
            = 1*(a[L][i]*(1-a[L][i]))*dC/da[L][i]