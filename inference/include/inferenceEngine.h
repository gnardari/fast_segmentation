#pragma once

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

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

#include <opencv2/opencv.hpp>

using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;

typedef struct EngConfig {
    unsigned int width;
    unsigned int height;
    unsigned int numClasses;
    string planPath;
    string inputTensorName;
    string outputTensorName;
} EngConfig;

class Logger : public ILogger           
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 };

class InferenceEngine{

    public:
        explicit InferenceEngine(EngConfig ec);
        ~InferenceEngine(void);
        void run(const cv::Mat& image, cv::Mat& classes);

    private:
        float* imageToTensor_(const cv::Mat &image);
        void argmax_(float *tensor, vector<unsigned char>& max);
        unsigned int countClasses_(float *tensor);

        string modelPath_;
        string inputTensorName_;
        string outputTensorName_;
        unsigned int width_;
        unsigned int height_;
        unsigned int numClasses_;
        unsigned int inputSizeBytes_;

        int inputBindIndex_;
        int outputBindIndex_;

        float* output_;
        float* inputDevice_;
        float* outputDevice_;
        float* bindings_[2];

        IRuntime* runtime_; 
        ICudaEngine* engine_;
        IExecutionContext *context_;
};
