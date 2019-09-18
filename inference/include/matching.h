#define PCL_NO_PRECOMPILE
// #include <Eigen/Eigen>
#include <pcl/registration/gicp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/geometry.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>

struct PointXYZD {
  PCL_ADD_POINT4D;

  float distances[8] = {10000,10000,10000,10000,10000,10000,10000,10000};

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZD,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, distances, distances)
)

//typedef pcl::PointCloud<pcl::PointXYZIR> PointCloudXYZIR;
typedef struct Match {

  PointXYZD a;
  PointXYZD b;

} Match;
