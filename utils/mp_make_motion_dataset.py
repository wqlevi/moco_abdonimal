from pathlib import Path
import sys, os
from functools import partial 
from glob import glob
from multiprocessing import Pool, current_process

from torchio.transforms import RandomMotion
import numpy as np
import nibabel as nib

def fileio(filename:str, action:RandomMotion, **kwargs):
    nii = nib.load(filename)
    arr = nii.get_fdata()
    out_arr = action(arr[None])[0]
    out_arr = np.where(out_arr >= 0, out_arr, 0)# 0-value cutoff
    new_name = nii.get_filename().replace("processed/","processed/sim/{}".format(kwargs['degrees'])).replace("raecker1","rawangq1").replace("/wat","_wat")
    new_nii = nib.Nifti1Image(out_arr, nii.affine)
    nib.save(new_nii, new_name)
    #print(new_name)

def main(filename:str, **kwargs):
    degrees = kwargs["degrees"] if "degrees" in kwargs else 1
    translation = kwargs["translation"] if "translation" in kwargs else 1
    rm = RandomMotion(degrees=degrees, translation=translation)
    fileio(filename, rm, **kwargs)
    


if __name__ == '__main__':
    data_path = sys.argv[1]
    degrees = translation = 5
    NUM_SUBJ:int=100
    kwargs = {'degrees': degrees,
              'translation': translation
              }

    filepaths = list(sorted(Path(data_path).glob("**/wat.nii.gz")))[:NUM_SUBJ]
    os.makedirs(Path(data_path.replace("raecker1","rawangq1"),'sim','{}'.format(kwargs['degrees'])), exist_ok=True)
    n_process = os.cpu_count()
    with Pool(n_process) as P:
        P.map(partial(main, **kwargs), filepaths)
        print(P)
        P.close()
        P.join()
    print("\033[93m Fertig \033[0m")
