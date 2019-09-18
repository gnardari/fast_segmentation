# Conversions

### checkpoint to .pb
```
TODO: get all that stuff as command line arg
cd conversions/scripts/
# fix paths, input and output nodes on convert_to_pb.py
python3 convert_to_pb.py
```

### .pb to uff
##### Needs to run a x86 machine, probably with TensorRT 5.1+

```
TODO: get all that stuff as command line arg
cd conversions/scripts/
# fix paths, input and output nodes on convert_pb_to_uff.py
python3 convert_pb_to_uff.py
```

### .uff to .plan
##### You need to do this conversion in the device that is going to use the model
```
cd conversions
# fix paths and file names on src/uff_to_plan.cpp
mkdir build && cd build
cmake ..
make -j4
cd src # inside build
./uff_to_plan models/erfnet_nobn.uff models/erfnet_nobnTX2.plan inputs/X 256 256 up23/BiasAdd 1 500000 float
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
