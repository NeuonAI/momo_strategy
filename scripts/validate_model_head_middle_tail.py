# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:22:45 2023

@author: user
"""

import pandas as pd
import os

TXT_FILE_LEN = "lists/class_file_len.txt"
CSV_PREDICTIONS = os.path.join(prediction_dir, "predictions_species.csv")


map_df = pd.read_csv(CSV_PREDICTIONS, delimiter=";")
map_groundtruth_ = map_df["groundtruth"].to_list()
map_groundtruth = [str(x) for x in map_groundtruth_]
map_top1 = map_df["top1 score"].to_list()
map_top5 = map_df["top5 score"].to_list()


with open(TXT_FILE_LEN, 'r') as f1:
    lines = [x for x in f1.readlines()]
txt_species = [x.split(" ")[0] for x in lines]
txt_filelen = [int(x.split(" ")[1]) for x in lines]


dictionary_category = {"head": [], "middle": [], "tail": []}
for spe, flen in zip(txt_species, txt_filelen):

    if flen < 31:
        dictionary_category["tail"].append(spe)
    if flen >= 31 and flen < 71:
        dictionary_category["middle"].append(spe)        
    if flen >= 71:
        dictionary_category["head"].append(spe)
        
#   Get head results
head_top1 = 0
head_top5 = 0
for spe in dictionary_category["head"]:
    spe_mapped_idx = map_groundtruth.index(spe)
    spe_top1 = int(map_top1[spe_mapped_idx])
    spe_top5 = int(map_top5[spe_mapped_idx])
    
    head_top1 += spe_top1
    head_top5 += spe_top5

dictionary_head_len = len(dictionary_category["head"])
print("Head Top-1:", round(head_top1 / dictionary_head_len,4), head_top1, "/", dictionary_head_len, " ----- Top-5:", round(head_top5 / dictionary_head_len,4), head_top5, "/", dictionary_head_len)


#   Get middle results
middle_top1 = 0
middle_top5 = 0
for spe in dictionary_category["middle"]:
    spe_mapped_idx = map_groundtruth.index(spe)
    spe_top1 = int(map_top1[spe_mapped_idx])
    spe_top5 = int(map_top5[spe_mapped_idx])
    
    middle_top1 += spe_top1
    middle_top5 += spe_top5

dictionary_middle_len = len(dictionary_category["middle"])
print("Middle Top-1:", round(middle_top1 / dictionary_middle_len,4), middle_top1, "/", dictionary_middle_len, " ----- Top-5:", round(middle_top5 / dictionary_middle_len,4), middle_top5, "/", dictionary_middle_len)



#   Get tail results
tail_top1 = 0
tail_top5 = 0
for spe in dictionary_category["tail"]:
    spe_mapped_idx = map_groundtruth.index(spe)
    spe_top1 = int(map_top1[spe_mapped_idx])
    spe_top5 = int(map_top5[spe_mapped_idx])
    
    tail_top1 += spe_top1
    tail_top5 += spe_top5

dictionary_tail_len = len(dictionary_category["tail"])
print("Tail Top-1:", round(tail_top1 / dictionary_tail_len,4), tail_top1, "/", dictionary_tail_len, " ----- Top-5:", round(tail_top5 / dictionary_tail_len,4), tail_top5, "/", dictionary_tail_len)
