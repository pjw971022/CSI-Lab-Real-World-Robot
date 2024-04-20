# export CC=/usr/bin/gcc-8
# export CXX=/usr/bin/g++-8
# export CUDAHOSTCXX=/usr/bin/gcc-8
# export CUDA_HOME=/usr/local/cuda-11.7 
# # # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib64:/usr/local/cuda/extras/CUPTI/lib64 
# export PATH=$PATH:$CUDA_HOME/bin 
# CUDACXX=/usr/local/cuda-11.7/bin/nvcc 
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.45 --no-cache-dir --force-reinstall

# set CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major" 
# set FORCE_CMAKE=1
python -m pip install llama-cpp-python==0.2.45 --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117 --no-cache-dir --force-reinstall
