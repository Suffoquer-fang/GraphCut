# Graph Cut

A Python implementation of Graph-Cut algorithm for texture synthesis, accelerated with FFT.

## Installation

Git clone this repo.
```shell
git clone https://github.com/suffoquer-fang/graphcut.git
```

Install the required packages.
```shell
pip install -r requirements.txt
```

## Quick Start
To generate texture images, use `patch_fitting` method.

```python
im_name = 'data/akeyboard_small.gif'
    
Gamma = 0.8
use_old_cut = True 
use_grad = False

im = Image.open(im_name).convert('RGB')
im_input = np.array(im, dtype=np.uint8)
h, w, _ = im_input.shape
height, width = 2 * h, 2 * w
im_src = np.zeros([height, width, _])
src_map = np.zeros([height, width]).astype(np.bool)
seam_map = SeamMap(height, width)

while not src_map.all():
    region_size = (h // 2, w // 2)
    error_region = get_error_region(im_src, src_map, seam_map, region_size)
    
    offset = get_offset_auto(im_src, src_map, im_input, error_region, region_size, Gamma)
    patch_fitting(im_src, src_map, im_input, offset, seam_map, use_old_cut, use_grad)

Image.fromarray(im_src.astype(np.uint8)).show()
```

## Example Results

### Different Offset Matching
Input texture --- Random Matching / Entire Matching / Subpatch Matching
![](https://suffoquer-fang/GraphCut/figs/offset-matching.jpeg)

### Old Seam
w/o old seam vs w/ old seam
![](https://suffoquer-fang/GraphCut/figs/old-seam.jpg)

### Surrounded Region
before patching vs after patching
![](https://suffoquer-fang/GraphCut/figs/surrounded.jpg)

### FFT Acceleration

| Time Cost (s) | Original | FFT-based |
| :------ | :------ | :------ |
| Entire Matching | 188.7 | 6.1 |
| Subpatch Matching | 62.3 | 4.2 |
| |
