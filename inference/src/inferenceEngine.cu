#include <inferenceEngine.h>

InferenceEngine::InferenceEngine(EngConfig ec){

    modelPath_ = ec.planPath;
    inputTensorName_ = ec.inputTensorName;
    outputTensorName_ = ec.outputTensorName;
    width_ = ec.width;
    height_ = ec.height;
    numClasses_ = ec.numClasses;
    inputSizeBytes_ = height_ * width_ * sizeof(float);

    ifstream planFile(modelPath_);
    stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    string plan = planBuffer.str();

    Logger gLogger;
    runtime_ = createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine((void*)plan.data(),
      plan.size(), nullptr);

    context_ = engine_->createExecutionContext();

    // Assuming output has the same height/width as the same as input
    cudaHostAlloc(&output_,
            inputSizeBytes_ * numClasses_, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&outputDevice_, output_, 0);

    inputBindIndex_ = engine_->getBindingIndex(inputTensorName_.c_str());
    outputBindIndex_ = engine_->getBindingIndex(outputTensorName_.c_str());

    bindings_[outputBindIndex_] = outputDevice_;
}

InferenceEngine::~InferenceEngine(void){
    cudaFree(inputDevice_);
    cudaFree(outputDevice_);

    cudaFreeHost(output_);

    engine_->destroy();
    context_->destroy();
    runtime_->destroy();
}
float* InferenceEngine::imageToTensor_(const cv::Mat & image)
{
  const unsigned int channels = image.channels();
  const unsigned int strides[3] = { height_ * width_, width_, 1 };

  float* tensor;
  cudaHostAlloc((void**)&tensor, channels*inputSizeBytes_, cudaHostAllocMapped);

  for (int i = 0; i < height_; i++) {
    const float* row_ptr = image.ptr<float>(i);
    for (int j = 0; j < width_; j++) {
        const unsigned int offset = i * strides[1] + j * strides[2];
        tensor[offset] = (float) row_ptr[j];
    }
  }
  return tensor;
}
/*
float* InferenceEngine::imageToTensor_(const cv::Mat & image)
{
  const unsigned int channels = image.channels();
  const unsigned int stridesCv[3] = { width_ * channels, channels, 1 };
  const unsigned int strides[3] = { height_ * width_, width_, 1 };

  float * tensor;
  cudaHostAlloc((void**)&tensor, channels*inputSizeBytes_, cudaHostAllocMapped);

  for (int i = 0; i < height_; i++) {
    for (int j = 0; j < width_; j++) {
      for (int k = 0; k < channels; k++) {
        const unsigned int offsetCv = i * stridesCv[0] + j * stridesCv[1] + k * stridesCv[2];
        const unsigned int offset = k * strides[0] + i * strides[1] + j * strides[2];
        tensor[offset] = (float) image.data[offsetCv];
      }
    }
  }

  return tensor;
}
*/
unsigned int InferenceEngine::countClasses_(float *tensor){
    // this function is just for debugging
    unsigned int totalA = 0;
    unsigned int totalB = 0;
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

void InferenceEngine::argmax_(float *tensor, vector<unsigned char>& max){
  unsigned int outSize = height_*width_*numClasses_;
  for (unsigned int i = 0; i < outSize; i += numClasses_){
      unsigned int maxIdx = i;
      unsigned char outIdx = 0;
      for(unsigned int c = 1; c < numClasses_; c++){
        if(tensor[maxIdx] < tensor[i+c]){
            maxIdx = i+c;
            outIdx = c;
        }
      }
      max.push_back(outIdx*255);
  }
}

void InferenceEngine::run(const cv::Mat & image, cv::Mat& out){
    timer_.startCpuTimer();
    float* input = imageToTensor_(image);
    timer_.endCpuTimer();
    std::cout << "input preproc: " << timer_.getCpuElapsedTimeForPreviousOperation() << std::endl;

    timer_.startGpuTimer();
    cudaHostGetDevicePointer(&inputDevice_, input, 0);
    bindings_[inputBindIndex_] = inputDevice_;
    timer_.endGpuTimer();
    std::cout << "bindings: " << timer_.getGpuElapsedTimeForPreviousOperation() << std::endl;

    timer_.startGpuTimer();
    context_->execute(1, (void**)bindings_);
    timer_.endGpuTimer();
    std::cout << "execute model: " << timer_.getGpuElapsedTimeForPreviousOperation() << std::endl;
    
    vector<unsigned char> max;
    //countClasses_(output_);
    argmax_(output_, max);

    //memcpy(out.data, input, 16*3600*sizeof(float));
    cudaFreeHost(input);

    memcpy(out.data, max.data(), max.size()*sizeof(unsigned char));
}
