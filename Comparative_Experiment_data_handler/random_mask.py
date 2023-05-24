import pandas as pd
import os
import argparse
import random
import copy
# from Comparative_Experiment.handler.tools.func import is_human_col
from tools import remove_front, is_vcf_format, is_human_col

# 按照mask_rate对vcf文件进行直接的随机掩蔽

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default=None)
parser.add_argument("-mr", "--mask_rate", type=float, default=0.4)
parser.add_argument("-o", "--out", type=str, default="masked.vcf")
args = parser.parse_args()

mask_rates = {}
mask_rates["origin_mask_rate"] = args.mask_rate
mask_rates["double_mask_rate"] = mask_rates["origin_mask_rate"] * mask_rates["origin_mask_rate"]
mask_rates["double_unmask_rate"] = (1 - mask_rates["origin_mask_rate"]) ** 2
mask_rates["left_mask_rate"] = (1 - mask_rates["double_mask_rate"] - mask_rates["double_unmask_rate"]) / 2
mask_rates["right_mask_rate"] = mask_rates["left_mask_rate"]
print(mask_rates)

def random_mask(genotype:str):
    r = random.random()
    if r < mask_rates["double_mask_rate"]:
        return ".|."
    elif r < (mask_rates["double_mask_rate"] + mask_rates["left_mask_rate"]):
        return f'.|{genotype[2]}'
    elif r < (mask_rates["double_mask_rate"] + mask_rates["left_mask_rate"] + mask_rates["right_mask_rate"]):
        return f'{genotype[0]}|.'
    return genotype

if is_vcf_format(args.file):
    # print("remove")
    remove_front(args.file, temp_file="./temp_mask")
    file = "./temp_mask"
else:
    file = args.file

df = pd.read_csv(file, sep='\t')
# print(df)
for col in df.columns:
    if is_human_col(col):
        df[col] = df[col].apply(random_mask)
# print(df)


df.to_csv(args.out, sep='\t', index=False, header=True)

os.system(f"sed -i \'1i\##fileformat=VCFv4.2\' {args.out}")
