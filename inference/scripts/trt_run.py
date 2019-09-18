import tensorrt as trt
import numpy as np
import pycuda.autoinit
from pycuda import driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
max_batch_size = 1
model_file = 'erfnet_nobn.uff'
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    # (num_channels, h, w)
    parser.register_input("inputs/X", (1, 256, 256))
    parser.register_output("up23/BiasAdd")
    parser.parse(model_file, network)

    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1 <<  20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
    with builder.build_cuda_engine(network) as engine:
        with engine.create_execution_context() as context:
            # h_input = cuda.pagelocked_zeros(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
            h_input = np.ones((1,256,256))
            h_output = cuda.pagelocked_zeros(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
            # Allocate device memory for inputs and outputs.
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            # Create a stream in which to copy inputs/outputs and run inference.
            stream = cuda.Stream()

            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(d_input, h_input, stream)
            # Run inference.
            context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            # Synchronize the stream
            stream.synchronize()
            # Return the host output. 
            output = h_output

output = output.reshape(256,256,2)
classes = np.argmax(output, axis=-1)
print(classes.shape)
print(np.bincount(classes.flatten()))
