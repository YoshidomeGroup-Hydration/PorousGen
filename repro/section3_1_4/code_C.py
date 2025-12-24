from scipy.stats import pearsonr
import numpy as np

img1 = np.load("results/seed_1/final_mask.npy")
img2 = np.load("results/seed_2/final_mask.npy")

corr, pval = pearsonr(img1.ravel(), img2.ravel())

print("Pearson correlation coefficient:", corr)
print("p-value:", pval)

with open("corr.txt", "w") as f:
    f.write(f"{corr}\n")
