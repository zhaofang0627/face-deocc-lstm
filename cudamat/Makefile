##################################
# Set path to dependencies.
# CUDA.
CUDA_ROOT=/usr/local/cuda-7.0
###################################

CUDA_LIB=$(CUDA_ROOT)/lib64
CUDA_BIN=$(CUDA_ROOT)/bin

NVCC = $(CUDA_BIN)/nvcc
FLAGS = -O3 --use_fast_math -v \
		    -gencode=arch=compute_20,code=sm_20 \
		    -gencode=arch=compute_30,code=sm_30 \
		    -gencode=arch=compute_35,code=sm_35 \
        -gencode=arch=compute_50,code=sm_50 \
        -gencode=arch=compute_52,code=sm_52 \
				--compiler-options '-fPIC' --shared -Xlinker -rpath -Xlinker $(CUDA_LIB) \
				-Xcompiler -rdynamic -lineinfo

all : libcudamat.so libcudamat_conv.so

libcudamat.so: cudamat.cu cudamat_kernels.cu cudamat.cuh cudamat_kernels.cuh
	$(NVCC) $(FLAGS) -o $@ cudamat.cu cudamat_kernels.cu -lcublas -L$(CUDA_LIB)
 
libcudamat_conv.so: cudamat_conv.cu cudamat_conv_kernels.cu cudamat_conv.cuh cudamat.cuh cudamat_conv_kernels.cuh
	$(NVCC) $(FLAGS) -o $@ cudamat_conv.cu cudamat_conv_kernels.cu -lcublas -L$(CUDA_LIB)

clean:
	rm -rf *.so
