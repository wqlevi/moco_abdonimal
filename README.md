# Understanding multi-modal representation for medical imaging via deep learning

## TODOS
- [x] load and reuse resnet50 from Veronika(2D data channel=1, across slices)
- [x] parallel compute TSNE on GPU, using [tsnecuda](https://github.com/CannyLab/tsne-cuda/tree/main)(only gpu-5, ___gpu-3 not working___)
    *   1 min 16 sec (CPU) vs 3 sec (CUDA)
    * GPU-03 reports architecture's error with faiss
- [ ] PCA, tSNE in data space and latent space of resnet50
- [ ] compare resnet50 with one of VLM from [awesome VLM](https://github.com/gokayfem/awesome-vlm-architectures)

- [ ] Trying Nvidia.DALI for faster dataloading, but dependencies `libnvjepg` is missing from cudatoolkit.12.0
