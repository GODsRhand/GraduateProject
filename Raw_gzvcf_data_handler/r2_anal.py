import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help="待计算r2的文件")
parser.add_argument('-r', '--range', type=int, help="待计算r2的索引范围")
parser.add_argument('-o', '--out', type=str, help="输出的图像")
args = parser.parse_args()

df = pd.read_csv(args.file, sep='\t')
cnt = args.range
out = args.out

r2 = df['R^2'].to_numpy()
mat = np.zeros((cnt, cnt))

n = 0
for x in range(cnt):
    for y in range(cnt):
        if x == y:
            mat[x][y] = 0
        elif x < y:
            mat[x][y] = r2[n]
            n += 1
        else:
            mat[x][y] = mat[y][x]

plt.matshow(mat, cmap=plt.cm.hot, vmin=mat.min(), vmax=mat.max())
plt.colorbar()
plt.savefig(out)
plt.show()
mat_pd = pd.DataFrame(mat)
mat_pd.to_csv('r2.csv')
print(mat_pd)