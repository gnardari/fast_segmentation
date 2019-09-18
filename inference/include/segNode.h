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
  // explicit Segmentation(ros::NodeHandle& nh, InferenceEngine& ie);
  explicit Segmentation(ros::NodeHandle& nh, EngConfig ec);

  Segmentation(const Segmentation&) = delete;
  Segmentation operator=(const Segmentation&) = delete;
  using Ptr = boost::shared_ptr<Segmentation>;
  using ConstPtr = boost::shared_ptr<const Segmentation>;

 private:
  void rangeCloudCb_(const sensor_msgs::ImageConstPtr& imgMsg,
                     const sensor_msgs::PointCloud2ConstPtr& cloudMsg);
  void preProcessRange_(cv::Mat& rImg, float maxDist);
  void extractSamples(cv::Mat& rImg, std::vector<cv::Mat>& samples);
  void maskCloud_(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& outCloud,
                  cv::Mat mask);
  ros::NodeHandle nh_;
  // InferenceEngine ie_;
  // image_transport::Subscriber rImgSub_;

  image_transport::ImageTransport it_;
  image_transport::Publisher rMaskPub_;
  image_transport::Publisher rImgPub_;
  ros::Publisher rawPubCloud_;
  ros::Publisher maskPubCloud_;
  pc_sub_type* sub_cloud_;
  image_sub_type* sub_img_;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::PointCloud2>
      ImgCloudPolicy;
  message_filters::Synchronizer<ImgCloudPolicy>* sync_;
};
}  // namespace trt_inference
