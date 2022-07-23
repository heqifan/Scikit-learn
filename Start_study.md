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
