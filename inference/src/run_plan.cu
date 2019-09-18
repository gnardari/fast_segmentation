#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>

// #include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

#include <opencv2/opencv.hpp>

// #include "logger.h"
// #include "common.h"

using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;

class Logger : public ILogger           
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 } gLogger;

float *imageToTensor(const cv::Mat & image)
{
  const size_t height = image.rows;
  const size_t width = image.cols;
  const size_t channels = image.channels();
  const size_t numel = height * width * channels;

  const size_t stridesCv[3] = { width * channels, channels, 1 };
  const size_t strides[3] = { height * width, width, 1 };

  float * tensor;
  cudaHostAlloc((void**)&tensor, numel * sizeof(float), cudaHostAllocMapped);

  for (int i = 0; i < height; i++) 
  {
    for (int j = 0; j < width; j++) 
    {
      for (int k = 0; k < channels; k++) 
      {
        const size_t offsetCv = i * stridesCv[0] + j * stridesCv[1] + k * stridesCv[2];
        const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
        tensor[offset] = (float) image.data[offsetCv];
      }
    }
  }

  return tensor;
}

/* float* createCudaBuffer(int height, int width, int num_classes, */
/*         string fname, bool map_mem){ */
/*   cv::Mat image = cv::imread(fname, 0); */
/*   cv::resize(image, image, cv::Size(width, height)); */
/*   float *input = imageToTensor(image); */
/*  */
/*   float *output; */
/*   float *inputDevice; */
/*   float *outputDevice; */
/*   size_t inputSize = height * width * image.channels() * sizeof(float); */
/*  */
/*   cudaHostAlloc(&output, num_classes * sizeof(float), cudaHostAllocMapped); */
/*  */
/*   if (map_mem){ */
/*     cudaHostGetDevicePointer(&inputDevice, input, 0); */
/*     cudaHostGetDevicePointer(&outputDevice, output, 0); */
/*   } */
/*   else { */
/*     cudaMalloc(&inputDevice, inputSize); */
/*     cudaMalloc(&outputDevice, num_classes * sizeof(float)); */
/*   } */
/*  */
/*   string input_tensor = "inputs/X"; */
/*   string output_tensor = "up23/BiasAdd"; */
/*   int inputBindIndex = engine->getBindingIndex(input_tensor.c_str()); */
/*   int outputBindIndex = engine->getBindingIndex(output_tensor.c_str()); */
/*   float *bindings[2]; */
/*   bindings[inputBindingIndex] = inputDevice; */
/*   bindings[outputBindingIndex] = outputDevice; */
/*   return &bindings; */
/* } */

size_t argmax(float *tensor, size_t numel)
{
  if (numel <= 0)
    return 0;

  size_t totalA = 0;
  size_t totalB = 0;
  vector<int> max;
  for (int i = 0; i < 16*3600*2; i+=2){
      if (tensor[i] < tensor[i+1]){
          max.push_back(1);
          totalA++;
      } else {
          max.push_back(0);
          totalB++;
      }
  }

  cout << "A: " << totalA << " B: " << totalB << endl;
  return 0;
}

void run(){
    IRuntime* runtime = createInferRuntime(gLogger);
    int num_runs = 1000;
    int width = 3600;
    int height = 16;
    int num_classes = 2;
    bool map_mem = true;
    string planPath = "/home/jetson/Documents/realtime_segmentation/models/simple_erfnet.plan";
    string imagePath = "/home/jetson/Documents/realtime_segmentation/data/input.jpg";
    string input_tensor = "inputs/X";
    string output_tensor = "up23/BiasAdd";

    cout << "Starting Inference" << endl;

    ifstream planFile(planPath);
    stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    string plan = planBuffer.str();

    ICudaEngine *engine = runtime->deserializeCudaEngine((void*)plan.data(),
      plan.size(), nullptr);
    IExecutionContext *context = engine->createExecutionContext();

    cout << "Initialized Engine" << endl;
    /*
    THIS SHOULD BE A FUNCTION
    */

    cv::Mat image = cv::imread(imagePath, 0);
    cv::resize(image, image, cv::Size(width, height));
    float *input = imageToTensor(image);

    cout << "Read Image" << endl;
    float *output;
    float *inputDevice;
    float *outputDevice;
    size_t inputSize = height * width * image.channels() * sizeof(float);

    cudaHostAlloc(&output, num_classes * sizeof(float), cudaHostAllocMapped);
    cout << "Memory Allocation" << endl;

    if (map_mem){
      cudaHostGetDevicePointer(&inputDevice, input, 0);
      cudaHostGetDevicePointer(&outputDevice, output, 0);
      cout << "Mapped GPU and host memory" << endl;
    }
    else {
      cudaMalloc(&inputDevice, inputSize);
      cudaMalloc(&outputDevice, num_classes * sizeof(float));
    }

    int inputBindIndex = engine->getBindingIndex(input_tensor.c_str());
    int outputBindIndex = engine->getBindingIndex(output_tensor.c_str());
    float *bindings[2];
    bindings[inputBindIndex] = inputDevice;
    bindings[outputBindIndex] = outputDevice;
    cout << "Got Bindings" << endl;

    //float *bindings = createCudaBuffer(256, 256, 2, imagePath, true);

    double avgTime = 0;

    for (int i = 0; i < num_runs + 1; i++){
        chrono::duration<double> diff;
        auto t0 = chrono::steady_clock::now();
        context->execute(1, (void**)bindings);
        //cout << "Ran execute" << endl;
        auto t1 = chrono::steady_clock::now();
        diff = t1 - t0;

    if (i != 0)
      avgTime += diff.count()*1000.0;
    }
    avgTime /= num_runs;
    cout << "Average inference time: " << avgTime << "ms" << endl;

    argmax(output, num_classes);

    cudaFree(inputDevice);
    cudaFree(outputDevice);

    cudaFreeHost(input);
    cudaFreeHost(output);

    engine->destroy();
    context->destroy();
    runtime->destroy();
}

int main(int argc, char * argv[]){
  run();
  return 0;
}
