from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import SoftInput

# downloading data
img = np.array(Image.open('sk.png'), dtype=np.uint8)

# random init
np.random.seed(25)
num = img.size // 30

np.random.seed(25)
row_ind = np.random.randint(0, img[:, : ,0].shape[0], size=num)
col_ind = np.random.randint(0, img[:, :, 0].shape[1], size=num)

# we assume that pixels at main diagonal are known to be sure that at least one
# pixel at each row and at each column is known
row_ind = np.append(row_ind, np.arange(img[:, :, 0].shape[0]))
col_ind = np.append(col_ind, np.arange(img[:, :, 0].shape[0]))

X_r = sparse.csr_matrix((img[(row_ind, col_ind, 0)], (row_ind, col_ind)), shape=img[:, :, 0].shape, dtype=np.float)
X_g = sparse.csr_matrix((img[(row_ind, col_ind, 1)], (row_ind, col_ind)), shape=img[:, :, 0].shape, dtype=np.float)
X_b = sparse.csr_matrix((img[(row_ind, col_ind, 2)], (row_ind, col_ind)), shape=img[:, :, 0].shape, dtype=np.float)

damaged_img = np.dstack((X_r.toarray(), X_g.toarray(), X_b.toarray()))
damaged_img = np.array(damaged_img, dtype=np.uint8)

np.random.seed(25)
si_r = SoftInput.SoftInput(X_r)
si_g = SoftInput.SoftInput(X_g)
si_b = SoftInput.SoftInput(X_b)

lambdas = np.linspace(0, 100, 10)

lambdas, appr_r_U, appr_r_Vt  = si_r.fit(lambdas=lambdas, maxiter=2000, start_rank=100, tol=1e-6)
lambdas, appr_g_U, appr_g_Vt = si_g.fit(lambdas=lambdas, maxiter=2000, start_rank=100, tol=1e-6)
lambdas, appr_b_U, appr_b_Vt = si_b.fit(lambdas=lambdas, maxiter=2000, start_rank=100, tol=1e-6)

Ans_r_si = np.array(appr_r_U[-1].dot(appr_r_Vt[-1]), dtype=np.uint8)
Ans_g_si = np.array(appr_g_U[-1].dot(appr_g_Vt[-1]), dtype=np.uint8)
Ans_b_si = np.array(appr_b_U[-1].dot(appr_b_Vt[-1]), dtype=np.uint8)

plt.figure(figsize=(16, 6))

plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.title('Original image', fontsize=20)

plt.subplot(132)
plt.imshow(damaged_img)
plt.axis('off')
plt.title('Damaged image', fontsize=20)

plt.subplot(133)
plt.imshow(np.dstack((Ans_r_si, Ans_g_si, Ans_b_si)))
plt.axis('off')
plt.title('Repaired image by Soft-Input', fontsize=20)

plt.savefig('demo.png')

