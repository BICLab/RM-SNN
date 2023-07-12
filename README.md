# Sparser Spiking Activity can be Better: Feature Refine-and-Mask Spiking Neural Network for Event-based Visual Recognition

#### Installation

1. Python 3.7.4
2. PyTorch 1.7.1
3. numpy 1.19.2
4. spikingjelly 0.0.0.0.12
5. nni 2.5

Please see `requirements.txt` for more requirements.

### **Example**

#### DVS128 Gesture

1. Download [DVS128 Gesture](https://www.research.ibm.com/dvsgesture/) and put the downloaded dataset to `/DVS128_Gesture/data`, then run `/DVS128_Gesture/data/DVS_Gesture`.py.

```
/DVS128_Gesture/
│  ├── /data/
│  │  ├── DVS_Gesture.py
│  │  └── DvsGesture.tar.gz
```

2. Change the values of T and dt in `/DVS128_Gesture/RM/Config.py` , `/DVS128_Gesture/CRM/Config.py` or `/DVS128_Gesture/TRM/Config.py` then run the tasks in `/DVS128` Gesture.

eg:

```
python RM_main.py
```

3. View the results in `/DVS128_Gesture/RM/Result/` 、 `/DVS128_Gesture/TRM/Result/` or `/DVS128_Gesture/CRM/Result/`.


### Extra

1. `/module/RM.py` defines RM,TRM,CRM layer and `/module/LIF.py`, `LIF_Module.py` defines LIF module.
2. Explain again if the parameters in the `Config.py` are different from the paper name

```
conifg.c_sparsity_ratio = 1 - beat_c
config.t_sparsity_ratio = 1 - beat_t
```

3. If it produces an error like

```
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
```

, please reduce the batch-size selected for this experimence.

> see https://github.com/pytorch/pytorch/issues/32564#issuecomment-635062872 for more info.
