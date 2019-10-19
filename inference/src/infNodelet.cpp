#include <infNodelet.h>
#include <pluginlib/class_list_macros.h>

namespace trt_inference {
void InferenceNodelet::onInit() {
  // ros::NodeHandle nh_(getMTPrivateNodeHandle());
  ros::NodeHandle nh_(getPrivateNodeHandle());

  const string planPath = "models/modeltrt.plan";
  const string input = "inputs/X";
  const string output = "up23/BiasAdd";

  EngConfig ec;
  ec.width = nh_.param("width", 900);
  ec.height = nh_.param("height", 16);
  ec.numClasses = nh_.param("numClasses", 2);
  ec.planPath = planPath;
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
  inference.reset(new Inference(getPrivateNodeHandle(), ec));
  ROS_INFO("Created SEGMENTATION Nodelet");
}
}  // namespace trt_inference

PLUGINLIB_EXPORT_CLASS(trt_inference::InferenceNodelet, nodelet::Nodelet)
