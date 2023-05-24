import argparse
import os
import pandas as pd
from tools import remove_front, is_vcf_format, add_front

# 对vcf源文件，从第begin个位点开始取出长度为length的片段
# 输出的文件是不带vcf文件格式注释的

parser = argparse.ArgumentParser()
parser.add_argument('-s', "--source", type=str, default="../temp/chr22.maf5e-2.recode.vcf", help="源文件")
parser.add_argument('-b', "--begin", type=int, default=512, help="起始位点")
parser.add_argument('-l', "--length", type=int, default=1024, help="切片长度")
parser.add_argument('-o', "--out", type=str, default="../temp", help="输出目录")
parser.add_argument('-p', "--prefix", type=str, default="out_site")
# parser.add_argument('-c', "--chrom_num", type=int, default=22, help="染色体序号，须为纯数字")
parser.add_argument('--temp', type=str, default="../temp/sites")
args = parser.parse_args()

if is_vcf_format(args.source):
    remove_front(src=args.source, temp_file=args.temp)
    file = args.temp
else:
    file = args.source

out_file = os.path.join(args.out, (args.prefix + f".{args.begin}_{(args.begin + args.length)}.vcf"))
df = pd.read_csv(file, sep='\t')

# df['#CHROM'] = df['#CHROM'].apply(lambda x:args.chrom_num)
# df['#CHROM'] = args.chrom_num

piece = df.iloc[args.begin: (args.begin + args.length)]
piece.to_csv(out_file, header=True, index=False, sep='\t')
print(piece)