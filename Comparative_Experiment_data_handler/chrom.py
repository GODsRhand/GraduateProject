import pandas as pd
import os
import argparse
import random
import copy
from tools import remove_front, is_vcf_format, add_front

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default=None)
parser.add_argument("-o", "--out", type=str, default=None)
parser.add_argument("-c", "--chr", type=int, default=22)
args = parser.parse_args()
temp_file = "./temp_chrom"

if is_vcf_format(args.file):
    remove_front(src=args.file, temp_file=temp_file)
    f = temp_file
else:
    f = args.file

print(f)
vcf = pd.read_csv(f, sep='\t')
vcf["#CHROM"] = args.chr
print(vcf)
vcf.to_csv(f, sep='\t', index=False)
add_front(f, temp_file=args.out)
