#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <opencv2/core/core.hpp>
#include <pcl_ros/point_cloud.h>
#include <iostream>
#include <random>
//#include <math.h>
#include <cmath>

using namespace boost;

struct TreeVertex {
  int treeId;
  double coords[3];
  std::vector<std::vector<double>> points;
};

using graph_t = adjacency_list<listS, vecS, directedS,
                TreeVertex, property<edge_weight_t, float>>;
using vertex_t = graph_traits<graph_t>::vertex_descriptor;
using edge_t   = graph_traits<graph_t>::edge_descriptor;

void graphToCloud(graph_t g, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
// Identifies arches and creates graph nodes based on centroids
std::vector<vertex_t> getVerticesFromBeam(const size_t rowIdx, const size_t maxColIdx,
        const unsigned char* depthBeam, const unsigned char* maskBeam,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, graph_t& graph);
double euclideanDist(double* vecA, double* vecB);
//Greedy matcher
void computeEdgesGreedy(const std::vector<vertex_t> beamAvertices,
    const std::vector<vertex_t> beamBvertices, graph_t& graph, float distThreshold);
void computeGraph(const cv::Mat rangeImg, const cv::Mat mask,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, graph_t& graph);
