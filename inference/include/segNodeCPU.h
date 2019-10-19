#pragma once

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <fstream>
#include <iostream>
// #include <inferenceEngine.h>
#include <inferenceEngineCPU.h>

typedef message_filters::Subscriber<sensor_msgs::Image> image_sub_type;
typedef message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub_type;
namespace imgEnc = sensor_msgs::image_encodings;

namespace trt_inference {
class Segmentation : public InferenceEngineCPU {
 public:
  explicit Segmentation(ros::NodeHandle& nh, EngConfig ec);

  Segmentation(const Segmentation&) = delete;
  Segmentation operator=(const Segmentation&) = delete;
  using Ptr = boost::shared_ptr<Segmentation>;
  using ConstPtr = boost::shared_ptr<const Segmentation>;

 private:
  void semanticSegCb_(const sensor_msgs::ImageConstPtr& imgMsg);
  void preProcessRange_(const cv::Mat& inpImg, cv::Mat& outImg);
  ros::NodeHandle nh_;

  image_transport::ImageTransport it_;
  image_transport::Publisher maskPub_;
  image_transport::Subscriber imgSub_;
};
}  // namespace trt_inference
