import csv
import os
import math
import numpy as np
from typing import List

in_file = r'C:\Users\BB\Desktop\training_data\nodule_eval\J_client0_test_annotation.csv'
out_file = r'C:\Users\BB\Desktop\training_data\nodule_eval\J_client0_test_seriesuid.csv'

class NoduleTyper:
    def __init__(self, image_spacing: List[float]):
        self.diamters = {'benign': [0,4], 
                        'probably_benign': [4, 6],
                        'probably_suspicious': [6, 8],
                        'suspicious': [8, -1]}
        
        self.spacing = np.array(image_spacing, dtype=np.float64)
        self.voxel_volume = np.prod(self.spacing)
        
        self.areas = {}
        for key in self.diamters:
            self.areas[key] = [round(self._compute_sphere_volume(self.diamters[key][0]) / self.voxel_volume),
                               round(self._compute_sphere_volume(self.diamters[key][1]) / self.voxel_volume)]
        
    def get_nodule_type_by_seg_size(self, nodule_size: float) -> str:
        for key in self.areas:
            if nodule_size >= self.areas[key][0] and (nodule_size < self.areas[key][1] or self.areas[key][1] == -1):
                return key
        return 'benign'

    def get_nodule_type_by_dhw(self, d: int, h: int, w: int) -> str:
        nodule_volume = self._compute_nodule_volume(w, h, d)
        return self.get_nodule_type_by_seg_size(nodule_volume)
    def _compute_sphere_volume(self,diameter: float) -> float:
        if diameter == 0:
            return 0
        elif diameter == -1:
            return 100000000
        else:
            radius = diameter / 2
            return 4/3 * math.pi * radius**3

    def _compute_nodule_volume(self,w: float, h: float, d: float) -> float:
        # We assume that the shape of the nodule is approximately spherical. The original formula for calculating its 
        # volume is 4/3 * math.pi * (w/2 * h/2 * d/2) = 4/3 * math.pi * (w * h * d) / 8. However, when comparing the 
        # segmentation volume with the nodule volume, we discovered that using 4/3 * math.pi * (w * h * d) / 8 results 
        # in a nodule volume smaller than the segmentation volume. Therefore, we opted to use 4/3 * math.pi * (w * h * d) / 6 
        # to calculate the nodule volume.
        volume = 4/3 * math.pi * ((w * h * d) / 6)
        return volume
    


def readCSV_getuid(filename):
    lines = []
    pre_id = ''
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            if line[0] == pre_id:
                continue
            pre_id = line[0]
            lines.append(line[0])
    return lines[1:]

def generate_seriesuid(in_file,out_file):
    print("start!!")  
    lines = readCSV_getuid(in_file)
    FN_list_file = open(os.path.join(out_file), 'w')
    for line in lines:
        FN_list_file.write(f"{line}\n")
    print("finish!!")

def gen_anno_noddule_type(in_file,out_file):
    print("start!!")  
    lines = readCSV_getuid(in_file)
    FN_list_file = open(os.path.join(out_file), 'w')
    for line in lines:
        FN_list_file.write(f"{line},nodule\n")
    print("finish!!")