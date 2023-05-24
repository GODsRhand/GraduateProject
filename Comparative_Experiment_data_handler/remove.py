import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help="待处理的文件")
parser.add_argument('-o', '--out', type=str, help="输出的csv文件")
args = parser.parse_args()

df = pd.read_csv(args.file, sep='\t')
ids = df[df['IN_FILE']=='B'][['CHROM', 'POS1']]
print(ids.shape)
ids.to_csv(args.out ,sep='\t', header=None, index=False)