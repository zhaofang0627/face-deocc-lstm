## Robust LSTM-Autoencoders for Face De-Occlusion in the Wild
Code for paper [Robust LSTM-Autoencoders for Face De-Occlusion in the Wild](https://ieeexplore.ieee.org/abstract/document/8101544) by Fang Zhao, Jiashi Feng, Jian Zhao, Wenhan Yang, Shuicheng Yan; TIP 2018.

### Getting Started
To compile cudamat library, modify `CUDA_ROOT` in `cudamat/Makefile` to the relevant cuda root path.

Install [caffe](https://github.com/BVLC/caffe) and [pycaffe](https://github.com/BVLC/caffe/tree/master/python).

Next compile .proto file by calling
```
protoc -I=./ --python_out=./ config.proto
```

### Training and Test
[lstm_ae_spatial_mr_com.py](https://github.com/zhaofang0627/face-deocc-lstm/blob/master/lstm_ae_spatial_mr_com.py): training and test for the RLA model described in the paper.

[lstm_ae_spatial_mr_com_ladv.py](https://github.com/zhaofang0627/face-deocc-lstm/blob/master/lstm_ae_spatial_mr_com_ladv.py): training and test for the Identity Preserving RLA (IP-RLA) model described in the paper.
