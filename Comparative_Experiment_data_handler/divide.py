import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-t', "--test_indvs", type=int, default=502, help="测试集的人数，默认值为502；剩余部分被划分为训练集")
parser.add_argument('-s', "--source", type=str, default="../temp/chr1.maf5e-2.recode.vcf", help="被划分的源文件")
parser.add_argument('-p', "--prefix", type=str, default="out", help="输出前缀")
parser.add_argument('-o', "--out", type=str, default="./", help="输出目录")
parser.add_argument('--temp', type=str, default="./", help="缓存文件夹")
args = parser.parse_args()
# 将vcf文件 划分为训练集和测试集两部分

train_prefix = os.path.join(args.out, args.prefix) + ".train"
test_prefix = os.path.join(args.out, args.prefix) + ".test"

# 划分测试集
print("划分测试集")
os.system(f"vcftools --vcf {args.source} --max-indv {args.test_indvs} --recode --out {test_prefix}")
os.system(f"vcftools --vcf {args.source} --diff {test_prefix}.recode.vcf --diff-indv --out diff_i")

# 寻找测试集和源文件indvs的交集
print("寻找测试集和源文件indvs的交集")
both = os.path.join(args.temp, "both_indv")
df = pd.read_csv("diff_i.diff.indv_in_files", sep='\t')
ids = df[df['FILES']=='B']['INDV']
ids.to_csv(both, sep='\t', header=None, index=False)

# 排除掉这个交集就是我们需要的训练集
print("划分训练集")
os.system(f"vcftools --vcf {args.source} --remove {both} --recode --out {train_prefix}")