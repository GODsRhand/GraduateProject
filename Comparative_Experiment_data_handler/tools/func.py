import os

def remove_front(src: str, temp_file: str="./temp"):
    """
    删除vcf文件头部的INFO，使其可以被当作csv文件读入成为dataframe
    """
    cnt = 0
    with open(src, "r") as f:
        # 寻找表头所在的行
        while(True):
            line = f.readline()
            if line.startswith("#CHROM") or line.startswith("CHROM"):
                break
            cnt += 1
        # 删除前cnt行
    # print(cnt)
    os.system(f"sed '1,{cnt}d' {src} > {temp_file}")

def add_front(src: str, temp_file: str="./temp"):
    """
    在vcf文件头部添加vcf格式相关的注释，使其可以被minimac处理
    """
    os.system(f"sed '1 i ##fileformat=VCFv4.2' {src} > {temp_file}")

def is_vcf_format(src: str) -> bool:
    """
    判断vcf文件开头是否注明了格式版本
    """
    with open(src, "r") as f:
        line = f.readline()
        return line.startswith("##fileformat=VCF")

def is_human_col(col_name: str) -> bool:
    if col_name.startswith("HG") or col_name.startswith("NA"):
        return True