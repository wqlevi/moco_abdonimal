from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import gc
import time
import sys

import torch

# Timing utilities
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()

    print(local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
    sys.stdout.flush()

@pipeline_def
def imagenet_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                    shard_id=shard_id,
                                    num_shards=num_shards,
                                    random_shuffle=is_training,
                                    pad_last_batch=True,
                                    name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                            device=decoder_device, output_type=types.RGB,
                                            device_memory_padding=device_memory_padding,
                                            host_memory_padding=host_memory_padding,
                                            preallocate_width_hint=preallocate_width_hint,
                                            preallocate_height_hint=preallocate_height_hint,
                                            random_aspect_ratio=[0.8, 1.25],
                                            random_area=[0.2, 1.0],
                                            num_attempts=100)
        images = fn.resize(images,
                        device=dali_device,
                        resize_x=crop,
                        resize_y=crop,
                        interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                device=decoder_device,
                                output_type=types.RGB)
        images = fn.resize(images,
                        device=dali_device,
                        size=size,
                        mode="not_smaller",
                        interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                    dtype=types.FLOAT,
                                    output_layout="CHW",
                                    crop=(crop, crop),
                                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                    mirror=mirror)
    labels = labels.gpu()
    return images, labels

if __name__ == '__main__':

    pipe = imagenet_dali_pipeline(batch_size=128,
                                    num_threads=8,
                                    device_id=0,
                                    seed=0,
                                    data_dir='animal',
                                    crop=224,
                                    size=256,
                                    dali_cpu=False,
                                    shard_id=0,
                                    num_shards=1,
                                    is_training=True)
    pipe.build()
    data_loader = DALIGenericIterator(pipe, ["img", "label"], reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP, auto_reset=True)

    start_timer()
    for data in data_loader:
        end_timer_and_print('data loading...')
        start_timer()
