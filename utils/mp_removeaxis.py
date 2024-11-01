from pathlib import Path
import sys, os
from functools import partial 
from glob import glob
from multiprocessing import Pool, current_process

import numpy as np
import nibabel as nib

    
def fileio(filename:str):
    nii = nib.load(filename)
    arr = nii.get_fdata()
    #out_arr = arr[0]
    out_arr = arr
    out_arr = np.where(out_arr >= 0, out_arr, 0) # 0-value cut-off
    assert out_arr.ndim == 3, "output must be 3D"
    new_name = nii.get_filename()
    new_nii = nib.Nifti1Image(out_arr, nii.affine)
    nib.save(new_nii, new_name)
    print(new_name)

def main(filename:str, **kwargs):
    fileio(filename)
    


if __name__ == '__main__':
    data_path = sys.argv[1]
    
    filepaths = list(sorted(Path(data_path).glob("*wat.nii.gz")))
    n_process = os.cpu_count()
    with Pool(n_process) as P:
        P.map(main, filepaths)
        print(P)
        P.close()
        P.join()
    print("\033[93m Fertig \033[0m")
