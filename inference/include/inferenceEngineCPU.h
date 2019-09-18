#include <fstream>
#include <boost/algorithm/string.hpp>

// #include "tensorflow/c/c_api.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <opencv2/opencv.hpp>

using namespace tensorflow;

typedef struct EngConfig {
    size_t width;
    size_t height;
    size_t numClasses;
    const char* modelPath;
    const char* inputTensorName;
    const char* outputTensorName;
} EngConfig;

class InferenceEngineCPU {
    public:
        explicit InferenceEngineCPU(EngConfig ec);
        ~InferenceEngineCPU();
        void run(const cv::Mat& rangeImg, cv::Mat& out);
        void preProcess(const cv::Mat& in, cv::Mat& out);
    protected:
        // config
        size_t width_;
        size_t height_;
        size_t numClasses_;
        std::string modelPath_;
        std::string inputTensorName_;
        std::string outputTensorName_;
    private:
        // tensorflow
        Session* session_;
        GraphDef graph_;
        Tensor is_training_;
        void argmax_(const float* in, std::vector<unsigned char>& max);
};
