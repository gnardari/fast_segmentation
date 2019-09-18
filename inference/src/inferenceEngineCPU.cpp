#include <inferenceEngineCPU.h>

InferenceEngineCPU::InferenceEngineCPU(EngConfig ec) {

    modelPath_ = std::string(ec.modelPath);
    inputTensorName_ = std::string(ec.inputTensorName);
    outputTensorName_ = std::string(ec.outputTensorName);
    width_ = ec.width;
    height_ = ec.height;
    numClasses_ = ec.numClasses;

    tensorflow::SessionOptions options = SessionOptions();
    // options.config.mutable_gpu_options()->set_allow_growth(true);
    Status status = NewSession(options, &session_);
    NewSession(options, &session_);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
    }

    //"segmentation/inference/models/tree_loam_4w_filter_cos.pb"
    // Read the protobuf graph
    ReadBinaryProto(Env::Default(), modelPath_, &graph_);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
    }

    // Add the graph to the session
    session_->Create(graph_);
    status = session_->Create(graph_);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
    }

    Tensor its_(DT_BOOL, TensorShape());
    its_.scalar<bool>()() = false;
    is_training_ = its_;
}

InferenceEngineCPU::~InferenceEngineCPU(void){
    // Free any resources used by the session
    session_->Close();
}

void InferenceEngineCPU::preProcess(const cv::Mat& in, cv::Mat& out){
  int d = 3;                                                                                                                                                                                                    
  double sigmaColor = 20;                                                                                                                                                                                       
  double sigmaSpace = 20;                                                                                                                                                                                       
  cv::BorderTypes borderType = cv::BORDER_ISOLATED;                                                                                                                                                             
  cv::bilateralFilter(in, out, d, sigmaColor, sigmaSpace,                                                                                                                                  
  borderType);                                                                                                                                                                              
}

void InferenceEngineCPU::argmax_(const float* in, std::vector<unsigned char>& max){
  size_t outSize = height_*width_*numClasses_;
  for (unsigned int i = 0; i < outSize; i += numClasses_){
      unsigned int maxIdx = i;
      unsigned char outIdx = 0;
      for(unsigned int c = 1; c < numClasses_; c++){
        if(in[maxIdx] < in[i+c]){
            maxIdx = i+c;
            outIdx = c;
        }
      }
      max.push_back(outIdx);
  }
}

void InferenceEngineCPU::run(const cv::Mat& rangeImg, cv::Mat& out){
    cv::Mat inp;
    preProcess(rangeImg, inp);

    int imgSize = inp.rows*inp.cols;
    // batch, flat img, channels
    Tensor rImgTensor(DT_FLOAT, TensorShape({1, inp.rows, inp.cols, 1}));
    // Tensor rImgTensor(DT_FLOAT, TensorShape({1, imgSize, 1}));
    StringPiece tmp_data = rImgTensor.tensor_data();
    memcpy(const_cast<char*>(tmp_data.data()), inp.data, imgSize*sizeof(float));

    std::vector<std::pair<string, Tensor>> feed_dict = {
        { inputTensorName_, rImgTensor }
        // { "is_training", is_training_ }
    };

    std::vector<tensorflow::Tensor> outputs;
    Status status = session_->Run(feed_dict, {outputTensorName_}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return;
    }
    
    // from tensor to array 
    float aux[inp.rows*inp.cols*2];
    memcpy(aux, outputs[0].flat<float>().data(), 2*imgSize*sizeof(float));

    // from array to cv::Mat
    std::vector<unsigned char> max;
    argmax_(aux, max);
    
    memcpy(out.data, max.data(), max.size()*sizeof(unsigned char));
    // cv::Mat aux(inp.rows, inp.cols, CV_32FC1, outputs[0].flat<float>().data());
    // std::cout << outputs[0].debugstring() << std::endl;
}
