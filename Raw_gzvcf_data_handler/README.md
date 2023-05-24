# 处理原始vcf数据时的脚本: Raw_gzvcf_data_handler

主要包括以下功能：
+ 从vcf文件生成训练用的csv文件`./gencsv_from_pieces.py`
+ 划分片段`pieces.py`
+ 从已划分的片段批量生成训练用的csv文件`gencsv_from_pieces.py`
+ 降低snp位点密度`density.py`
+ 连锁不平衡关系分析`r2_anal.py`

