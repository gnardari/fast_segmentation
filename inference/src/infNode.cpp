#include <infNode.h>

namespace trt_inference {
Inference::Inference(ros::NodeHandle& nh, EngConfig ec)
    : nh_(nh), it_(nh), InferenceEngine(ec) {
  
  // Publisher
  maskPub_ = it_.advertise("maskImg", 10);
  // Subscriber
  imgSub_ = it_.subscribe("image", 1, &Inference::semanticSegCb_, this);
}

void Inference::preProcessRange_(const cv::Mat& inpImg, cv::Mat& outImg) {
    // you can add preprocessing steps here
    outImg = inpImg.clone();
}

void Inference::semanticSegCb_(
    const sensor_msgs::ImageConstPtr& imgMsg) {

  ROS_INFO("Got new msg");
  try {
    cv::Mat rawImg = cv_bridge::toCvShare(imgMsg)->image;
    cv::Mat ppImg;
    preProcessRange_(rawImg, ppImg);

    // get nn output
    ROS_INFO("Running inference");
    cv::Mat mask = cv::Mat::zeros(height_, width_, CV_8U);
    run(ppImg, mask);

    maskPub_.publish(
        cv_bridge::CvImage(imgMsg->header, imgEnc::MONO8, mask).toImageMsg());
    
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'mono16'.",
              imgMsg->encoding.c_str());
  }
}
}  // namespace trt_inference
