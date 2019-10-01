#include <inferenceEngine.h>
#include <chrono>

int main(int argc, char** argv){
  string planPath = "/home/gnardari/Documents/dd/ag/tensorflow_segmentation/models/tree_loam_4w_filter_cos.plan";
  string inpPath = "/root/bags/models/input1.png";
  string inpName = "inputs/X";
  string outName = "up23/BiasAdd";

  EngConfig ec;
  ec.width = 900;
  ec.height = 16;
  ec.numClasses = 2;
  ec.planPath = planPath;
  ec.inputTensorName = inpName;
  ec.outputTensorName = outName;

  InferenceEngine eng(ec); 
  cv::Mat inp = cv::imread(inpPath, CV_32F);
  cv::Mat mask(16, 900, CV_8U);

  int num_runs = 11;
  double avgTime = 0;

  for (int i = 0; i < num_runs + 1; i++){
    std::chrono::duration<double> diff;
    auto t0 = std::chrono::steady_clock::now();
    eng.run(inp, mask);
    //cout << "Ran execute" << endl;
    auto t1 = std::chrono::steady_clock::now();
    diff = t1 - t0;

   if (i != 0)
     avgTime += diff.count()*1000.0;
   }
   avgTime /= num_runs;
   std::cout << "Average inference time: " << avgTime << "ms" << std::endl;
   cv::imwrite("test.png", mask);
}

// ec.width = nh_.param("width", 3600);
// ec.height = nh_.param("height", 16);
// ec.numClasses = nh_.param("numClasses", 2);
// ec.modelPath = nh_.param("modelPath", modelPath);
// ec.inputTensorName = nh_.param("inputTensorName", input);
// ec.outputTensorName = nh_.param("outputTensorName", output);
