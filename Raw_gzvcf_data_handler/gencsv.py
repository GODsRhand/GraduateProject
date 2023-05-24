import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--file", type=str)
parser.add_argument('-o', "--out", type=str)
args = parser.parse_args()



os.system(f"vcftools --vcf {args.file} --IMPUTE --out {args.out}_subset")
legend = pd.read_csv(f"{args.out}_subset.impute.legend", sep=' ')
ref_alt = []
ref = legend['allele0'].to_numpy()
alt = legend['allele1'].to_numpy()
hap = pd.read_csv(f"{args.out}_subset.impute.hap", sep=' ', header=None).T
hap = hap.to_numpy()
hap = hap * alt + (1 - hap) * ref
hap = hap.tolist()
hap = pd.DataFrame(hap)
hap.to_csv(f"{args.out}.csv", header=None, index=None)
print(f"{args.out}共{hap.shape[0]}个样本，每个样本包含{len(ref)}个SNP(s)")
        
