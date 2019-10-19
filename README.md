# TODOs:

- Add more documentation
- Make inference more generic (command line/launch file configs)
- Change the way the input is passed to the gpu in InferenceEngine.cu
- Add launch files

# Dependencies
1. Install nvidia drivers/cuda
2. Install Tensorflow ```pip install tensorflow or pip install tensorflow-gpu```
3. Install TensorRT ``` sudo aptitude install libnvinfer5 libnvinfer-dev```

## Conversions

`(outputNodeName)+` means one or more names

### checkpoint to .pb
```
cd conversions/scripts/
python3 convert_to_pb.py path/to/model.chk.meta path/to/model.chk.data path/to/output.pb (outputNodeName)+
```

### .pb to uff
##### Needs to run a x86 machine, probably with TensorRT 5.1+

```
cd conversions/scripts/
python3 convert_pb_to_uff.py path/to/model.pb path/to/output.uff (outputNodeName)+
```

### .uff to .plan
##### You need to do this conversion on the device that is going to use the model
```
cd conversions
mkdir build && cd build
cmake ..
make -j4
cd src # inside build
./uff_to_plan models/mymodel.uff models/mymodel.plan inputs/X 256 256 1 up23/BiasAdd 1 500000 float
```

# Inference

```
cd inference
# fix paths and file names on src/run_plan.cu
mkdir build && cd build
cmake ..
make -j4
cd src # inside build
./run_plan
```


This project uses scripts from [NVIDIA trt_image_classification](https://github.com/NVIDIA-AI-IOT/tf_to_trt_image_classification)
