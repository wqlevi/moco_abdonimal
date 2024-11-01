from PIL import Image
import numpy as np
from glob import glob

from multiprocessing import Pool

def print_stats(filename:str):
    arr = np.array(Image.open(filename))
    return arr.mean(), arr.std()

filelist = sorted(glob("/mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/2d/sim/*.png"))


def main(filename:str):
    m,s = 0,0
    for f in filelist:
         rst = print_stats(f)
         m += rst[0]
         s += rst[1]
    m /= len(filelist)
    s /= len(filelist)

    print(m,s)
main(filelist)
"""
    

n_process = 10
mean , std = 0,0
with Pool(n_process) as P:
    rst = P.map(print_stats, filelist)
    for r in rst:
        mean += r[0]
        std += r[1]
    #print(P)
    P.close()
    P.join()
mean /= 10
std /= 10
print(mean, std)
"""
