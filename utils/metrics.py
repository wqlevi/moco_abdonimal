import numpy as np

from skimage.metrics import structural_similarity, normalized_root_mse

def ssim(image:np.ndarray,gt:np.ndarray):
    '''
    input
    -----
        image: numpy.ndarray [B,C,H,W]
        gt: numpy.ndarray
    output
    -----
        numpy scalar of ssim averaged for channels
    '''
    #if not (isinstance(image,np.ndarray) and isinstance(gt,np.ndarray)):
    #    raise ValueError(f"both inputs should be in numpy.ndarray type, got {type(image)} and {type(gt)}")
    if not image.ndim == gt.ndim:
        raise ValueError("dimensiom of the inputs should be the same")

    data_range = np.max(gt) - np.min(gt)
    if image.ndim==4: # [B,C,H,W] 2D Tensor   # FIXME: calculation seems to be wrong
        ssim = 0
        for ch in range(gt.shape[-1]):
            ssim += structural_similarity(
                    image[ch].transpose(1,2,0),
                    gt[ch].transpose(1,2,0),
                    multichannel=True,
                    channel_axis=-1,
                    data_range=data_range
                    ) 
        ssim /= ch
        return ssim

        #return structural_similarity(image.transpose(1,2,3,0), gt.transpose(1,2,0), data_range = data_range, multichannel=True)
    elif image.ndim==3: # H,W,L Batch_size = 1
        return structural_similarity(image, gt, data_range = data_range, channel_axis=-1)

def psnr(image:np.ndarray,gt:np.ndarray)->np.ndarray:
    mse = np.mean((image - gt)**2)
    if mse == 0:
        return float('inf')
    #    data_range = np.max(gt) - np.min(gt)
    data_range= gt.max() - gt.min() # choose 1 if data in float type, 255 if data in int8
    return 20* np.log10(data_range) - 10*np.log10(mse)
