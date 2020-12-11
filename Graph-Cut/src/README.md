## Graph Cut

### 使用
在`main.py`中修改

设置输入图片
```python
im_name = 'data/akeyboard_small.jpg'
```

设置块偏移生成算法
```python
place_method = 'auto' # random, entire, subpatch, auto
```

设置使用old seam
```python
use_old_cut = True 
```

设置使用带梯度的损失函数
```python
use_grad = True 
```

设置策略参数$\Gamma$
```python
Gamma = 0.8
```

运行
```shell
python main.py
```