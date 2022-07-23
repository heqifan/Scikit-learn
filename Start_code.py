#NumPy
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))


#Scipy
from scipy import sparse

#scikit-learn 利用SciPy 中的函数集合来实现算法。对我们来说，SciPy 中最重要的是scipy.sparse：它可以给出稀疏矩阵 sparse matrice）.
#稀疏矩阵是scikit-learn 中数据的另一种表示方法。如果想保存一个大部分元素都是0 的二维数组，就可以使用稀疏矩阵：
#创建一个二维NumPy数组，对角线为1，其余都为0

eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

# 将NumPy数组转换为CSR格式的SciPy稀疏矩阵
# 只保存非零元素
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

# 通常来说，创建稀疏数据的稠密表示（dense representation）是不可能的（因为太浪费内
# 存），所以我们需要直接创建其稀疏表示（sparse representation）。下面给出的是创建同一
# 稀疏矩阵的方法，用的是COO 格式：
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo))

