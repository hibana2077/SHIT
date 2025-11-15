# TinyNet ImageNet Classification Performance

| model | img_size | top1 | top1_err | top5 | top5_err | param_count | crop_pct | interpolation |
|-------|----------|------|----------|------|----------|-------------|----------|----------------|
| tinynet_a.in1k | 192 | 77.666 | 22.334 | 93.532 | 6.468 | 6.19 | 0.875 | bicubic |
| tinynet_b.in1k | 188 | 74.946 | 25.054 | 92.194 | 7.806 | 3.73 | 0.875 | bicubic |
| tinynet_c.in1k | 184 | 71.214 | 28.786 | 89.746 | 10.254 | 2.46 | 0.875 | bicubic |
| tinynet_d.in1k | 152 | 66.936 | 33.064 | 87.078 | 12.922 | 2.34 | 0.875 | bicubic |
| tinynet_e.in1k | 106 | 59.874 | 40.126 | 81.774 | 18.226 | 2.04 | 0.875 | bicubic |

## Results on CUB-200-2011

## Results on Cotton80

- FVCORE for calculating FLOPS:

| code | model | top1 acc | gflops |
|------|-------|----------|----------|
| TNACT001 | TinyNet-A | 0.3583 | 0.3465 |
| TNBCT001 | TinyNet-B | TBC | TBC |
| TNCCT001 | TinyNet-C | TBC | TBC |
| TNDCT001 | TinyNet-D | TBC | TBC |
| TNECT001 | TinyNet-E | 0.2542 | 0.02457 |