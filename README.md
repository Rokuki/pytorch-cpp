# pytorch-cpp
## Usage
Download libtorch
```
$ wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
$ unzip libtorch-shared-with-deps-latest.zip
```
Tracing model and save model.pt
1. Modify to_cpp.py
2. Generate model.pt
```
python to_cpp.py
```

Build
```
$ cd pytorch-cpp
$ mkdir build;cd build
$ cmake -DCMAKE_PREFIX_PATH=../libtorch ..
$ make
$ ./example-app ../model.pt <img_path>
```
![output](https://github.com/Rokuki/pytorch-cpp/blob/master/output.png)
