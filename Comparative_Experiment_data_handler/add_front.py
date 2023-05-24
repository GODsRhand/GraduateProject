import argparse
import os
import pandas as pd
from tools import remove_front, is_vcf_format, add_front

parser = argparse.ArgumentParser()
parser.add_argument('-s', "--source", type=str, default="../temp/chr22.maf5e-2.recode.vcf", help="源文件")
parser.add_argument('-o', "--out", type=str, default="../temp", help="输出目录")
args = parser.parse_args()

if not is_vcf_format(args.source):
    add_front(src=args.source, temp_file=args.out)
