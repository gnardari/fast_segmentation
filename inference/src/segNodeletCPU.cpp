#include <segNodeletCPU.h>
#include <pluginlib/class_list_macros.h>

namespace trt_inference {
void SegmentationNodelet::onInit() {
  // ros::NodeHandle nh_(getMTPrivateNodeHandle());
  ros::NodeHandle nh_(getPrivateNodeHandle());

  const char* modelPath = "/root/bags/models/tree_loam_4w_alldata_maxn.pb";
  const char* input = "inputs/X";
  const char* output = "up23/BiasAdd";

  EngConfig ec;
  ec.width = nh_.param("width", 900);
  ec.height = nh_.param("height", 16);
  ec.numClasses = nh_.param("numClasses", 2);
  ec.modelPath = modelPath;
  ec.inputTensorName = input;
  ec.outputTensorName = output;
  // ec.modelPath = nh_.param<const char*>("modelPath", modelPath);
  // ec.inputTensorName = nh_.param<const char*>("inputTensorName", input);
  // ec.outputTensorName = nh_.param<const char*>("outputTensorName", output);
  /*
  nh_.getParam("width", ec.width);
  nh_.getParam("height", ec.height);
  nh_.getParam("numClasses", ec.numClasses);
  nh_.getParam("planPath", ec.planPath);
  nh_.getParam("inpTensorName", ec.inputTensorName);
  nh_.getParam("outTensorName", ec.outputTensorName);
  */
  segmentation.reset(new Segmentation(getPrivateNodeHandle(), ec));
  ROS_INFO("Created SEGMENTATION Nodelet");
}
}  // namespace trt_inference

PLUGINLIB_EXPORT_CLASS(trt_inference::SegmentationNodelet, nodelet::Nodelet)
