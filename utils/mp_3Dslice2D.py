from pathlib import Path
import sys, os
from functools import partial 
from glob import glob
from multiprocessing import Pool, current_process
from functools import partial

import nibabel as nib
import numpy as np
from PIL import Image

norm = lambda x: (x-x.min())*255/(x.max() - x.min())
convert_im = lambda x: Image.fromarray(x).convert('L')
cut_negative_values = lambda x: np.where(x>=0, x, 0)
def inner_loop(arr, *xyz, filename:str="", **kwargs):
    """
    input:
        2d array: np.ndarray
        i,j,k index along 3 axis: List[int]
        single filename: str
    usage:
        convert array to PIL image in mode:'L'
        change filename and save
    """
    i, ax = xyz
    im = convert_im(arr)
    filename_save = filename.replace(innest_folder,"2d/"+innest_folder).replace(kwargs['filename_pattern'], "_wat_ax-{}={}.png".format(ax, i))
    #print(filename_save)
    im.save(filename_save)
    
def process_array(arr:np.ndarray, filename, **kwargs):
    """
    input:
        3D array
        filename of the nii:str
    usage:
        * normalize 3D array [0,255]
        * iterate through 3 axis and slice them into 2Ds, naming `{filename}_wat_ax-{ax}={i}.png`
    """
    arr = cut_negative_values(arr) if arr.min() < 0 else arr # negative value cut-off
    arr = norm(arr) # norm between [0, 255]
    [inner_loop(arr.transpose(t,t-2,t-1)[i], i, t, filename=filename, **kwargs) for t in range(3) for i in range(arr.shape[t])] # transpose to permute axis for saving

    
def fileio(filename:str, **kwargs):
    nii = nib.load(filename)
    arr = nii.get_fdata()
    file_name = nii.get_filename()
    assert arr.ndim == 3, "3D input expected!"
    process_array(arr, file_name, **kwargs)

def main(filename:str, **kwargs):
    fileio(filename, **kwargs)
    


if __name__ == '__main__':
    data_path = sys.argv[1] if not sys.argv[1].endswith("/") else sys.argv[1][:-1]

    global innest_folder
    innest_folder = data_path.rsplit("/",1)[-1]

    # --- kwargs dict --- #
    kwargs = {}
    kwargs["wildcard_pattern"] = "*wat.nii.gz" if innest_folder == "sim" else "**/wat.nii.gz"
    kwargs["filename_pattern"] = "_wat.nii.gz" if innest_folder == "sim" else "/wat.nii.gz"

    filepaths = list(sorted(Path(data_path).glob(kwargs['wildcard_pattern'])))[:100]
    print("\033[93m In Total: %d files \033[0m"%(len(filepaths)))
    os.makedirs(data_path.replace(innest_folder,'2d/'+innest_folder), exist_ok=True)

    n_process = os.cpu_count()
    with Pool(n_process) as P:
        P.map(partial(main, **kwargs), filepaths)
        print(P)
        P.close()
        P.join()
    print("\033[93m Fertig \033[0m")
