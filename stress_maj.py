import networkx as nx
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt

EPSILON = 0.000001

def stress(X, weights, distances, n):
	print "Parameters:", X, type(X)
	s = 0.0
	for i in range(0,n):
		for j in range(i+1,n):
			norm = X[i,:] - X[j,:]
			norm = np.sqrt(sum(pow(norm,2)))
			s += weights[i,j] * pow((norm - distances[i,j]), 2)
	return s


# G = nx.petersen_graph()
G = nx.complete_graph(7)
# G = nx.wheel_graph(5)

n = G.number_of_nodes()
distances = nx.floyd_warshall(G)

X_curr = np.random.rand(n,2)*50 - 50
Z=np.copy(X_curr)
X_prev = np.copy(X_curr)

distances = np.array([[distances[i][j] for j in distances[i]] for i in distances])
weights = 1/pow(distances,2)

L = -weights
weights[weights == inf] = 0
L[L==-inf] = 0

diagL = np.diag(np.sum(weights, axis = 1))
L = L + diagL

delta = np.multiply(weights,distances)

diag_1s = np.repeat(-1,n)
diag_1s = np.diag(diag_1s)

col1s_mat = np.zeros([n,n])

LZ = np.zeros([n,n])


X_prev = np.copy(X_curr)
Z = np.copy(X_curr)
LZ = np.zeros([n,n])

for i in range(0, n):
	col1s_mat_ith = np.copy(col1s_mat)
	col1s_mat_ith[:,i] = 1
	col1s_mat_ith = col1s_mat_ith + diag_1s

	LZ_ith_row = np.matmul(col1s_mat_ith,Z)
	LZ_ith_row = pow(LZ_ith_row, 2)
	LZ_ith_row = np.sum(LZ_ith_row, axis = 1)
	LZ_ith_row = np.sqrt(LZ_ith_row)
	LZ_ith_row = 1. / LZ_ith_row
	LZ_ith_row[LZ_ith_row== -inf] = 0
	LZ_ith_row[LZ_ith_row == inf] = 0

	LZ_ith_row = LZ_ith_row.transpose();

	LZ[i,:] = LZ_ith_row

LZ = -LZ
diagLZ = -np.diag(np.sum(LZ, axis = 1))
LZ = LZ + diagLZ
# Now that the LZ matrix is constructed
# Time to solve for X

b = np.matmul(LZ, Z)
X_curr =  np.linalg.lstsq(L, b)[0]
# X_curr =  np.linalg.solve(L, b)

# Plot the X values on every iteration
# Or store the X values and replay later
print(X_curr)

while((stress(X_prev, weights, distances, n) - stress(X_curr, weights, distances, n))/stress(X_prev, weights, distances, n) >= EPSILON):

	X_prev = np.copy(X_curr)
	Z = np.copy(X_curr)
	LZ = np.zeros([n,n])

	for i in range(0, n):
		col1s_mat_ith = np.copy(col1s_mat)
		col1s_mat_ith[:,i] = 1
		col1s_mat_ith = col1s_mat_ith + diag_1s

		LZ_ith_row = np.matmul(col1s_mat_ith,Z)
		LZ_ith_row = pow(LZ_ith_row, 2)
		LZ_ith_row = np.sum(LZ_ith_row, axis = 1)
		LZ_ith_row = np.sqrt(LZ_ith_row)
		LZ_ith_row = 1. / LZ_ith_row
		LZ_ith_row[LZ_ith_row== -inf] = 0
		LZ_ith_row[LZ_ith_row == inf] = 0

		LZ_ith_row = LZ_ith_row.transpose();

		LZ[i,:] = LZ_ith_row

	LZ = -LZ
	diagLZ = -np.diag(np.sum(LZ, axis = 1))
	LZ = LZ + diagLZ
	# Now that the LZ matrix is constructed
	# Time to solve for X

	b = np.matmul(LZ, Z)
	X_curr =  np.linalg.lstsq(L, b)[0]
	# X_curr =  np.linalg.solve(L, b)

	# Plot the X values on every iteration
	# Or store the X values and replay later
	print(X_curr)
	
print(X_curr)
plt.scatter(X_curr[:,0], X_curr[:,1])
plt.show()
















