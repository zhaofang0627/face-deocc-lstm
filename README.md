## Robust LSTM-Autoencoders for Face De-Occlusion in the Wild
Code for paper [Robust LSTM-Autoencoders for Face De-Occlusion in the Wild](https://ieeexplore.ieee.org/abstract/document/8101544) by Fang Zhao, Jiashi Feng, Jian Zhao, Wenhan Yang, Shuicheng Yan; TIP 2018.

To compile cudamat library you need to modify `CUDA_ROOT` in `cudamat/Makefile` to the relevant cuda root path.

To compile .proto file you need to call
```
protoc -I=./ --python_out=./ config.proto
```

[lstm_ae_spatial_mr_com.py](https://github.com/zhaofang0627/face-deocc-lstm/blob/master/lstm_ae_spatial_mr_com.py): training and test for the RLA model described in the paper.

[lstm_ae_spatial_mr_com_ladv.py](https://github.com/zhaofang0627/face-deocc-lstm/blob/master/lstm_ae_spatial_mr_com_ladv.py): training and test for the Identity Preserving RLA (IP-RLA) model described in the paper.
