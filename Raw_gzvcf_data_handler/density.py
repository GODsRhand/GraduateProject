import argparse
import os
import random

# 按ratio抽取snp位点

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help="待处理的文件")
parser.add_argument('-r', '--ratio', type=float, help="抽取snp位点的概率，在0到1之间")
parser.add_argument('-o', '--out', type=str, help="输出文件所在路径")
args = parser.parse_args()

ratio = args.ratio
out = os.path.join(args.out, f"density_{ratio}.vcf")

with open(args.file) as vcf:
    with open(out, "w+") as o:
        o.write(vcf.readline())
        while True :
            line = vcf.readline()
            if line :
                if random.random() < ratio:
                    o.write(line)
            else:
                break
    