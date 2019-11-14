# TODOs:

- Make inference more generic (command line/launch file configs)
- Add time benchmarks

# Dependencies
1. Install nvidia drivers/cuda
2. Install Tensorflow ```pip install tensorflow or pip install tensorflow-gpu```
3. Install TensorRT ``` sudo aptitude install libnvinfer5 libnvinfer-dev```

## Conversions

`(outputNodeName)+` means one or more names

### checkpoint to .pb
```
cd conversions/scripts/
python3 convert_to_pb.py path/to/model.chk.meta dir/of/checkpoint/ path/to/output.pb (outputNodeName)+
```

### .pb to uff
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

# Example Model

We provide a simple model trained on [Kitti Road detection](http://www.cvlibs.net/datasets/kitti/eval_road.php) that can be used to test the entire pipeline [here](https://drive.google.com/drive/folders/12T8LE0TrVuoZUSMmXvwAZeWKNbiDMTqt?usp=sharing). All inputs were resized to (368x1200) for training.

**Input**
![Input](https://raw.githubusercontent.com/gnardari/fast_segmentation/master/data/input.png)

**Output**
![Output](https://raw.githubusercontent.com/gnardari/fast_segmentation/master/data/out.png)

This project uses scripts from [NVIDIA trt_image_classification](https://github.com/NVIDIA-AI-IOT/tf_to_trt_image_classification)
