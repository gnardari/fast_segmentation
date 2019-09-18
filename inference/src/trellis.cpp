#include <trellis.h>

void graphToCloud(graph_t g, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud){
    size_t numPoints = 0;
    std::vector<double> color_values(num_vertices(g));
   
    std::iota(std::begin(color_values), std::end(color_values), 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(color_values.begin(), color_values.end(), gen);

    for(auto vd : make_iterator_range(vertices(g))){
        if(g[vd].treeId == -1) continue;
        for(auto coord : g[vd].points){
            pcl::PointXYZI pclP;
            pclP.x = coord[0];
            pclP.y = coord[1];
            pclP.z = coord[2];
            pclP.intensity = color_values[g[vd].treeId];
            cloud->push_back(pclP);
            numPoints++;
        }
    }
    cloud->width = numPoints;
    cloud->height = 1;
    cloud->is_dense = false;
}

// Identifies arches and creates graph nodes based on centroids
std::vector<vertex_t> getVerticesFromBeam(const size_t rowIdx, const size_t maxColIdx,
        const unsigned char* depthBeam, const unsigned char* maskBeam,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, graph_t& graph){

  float xmean = 0;
  float ymean = 0;
  float zmean = 0;
  size_t numPoints = 0;
  size_t idx = 0;
  size_t next = 0;
  std::vector<vertex_t> vertices;
  std::vector<int> vertexIndices;
  std::vector<std::vector<double>> vertexPoints;
  for(size_t i=0; i<maxColIdx; ++i){
    // if we are in the last column, compare it to first
    if(i == maxColIdx-1){
      next = 0;
    } else {
      next = i+1;
    }
    if(maskBeam[i] == 1 && depthBeam[i]-depthBeam[next] < 0.5){
      vertexIndices.push_back(i);
    } else {
        xmean = 0;
        ymean = 0;
        zmean = 0;
        numPoints = 0;
        while (!vertexIndices.empty()){ 
          idx = vertexIndices.back(); 
          pcl::PointXYZI p = cloud->at(idx, rowIdx);

          if(pcl::isFinite(p)){
              xmean += p.x;
              ymean += p.y;
              zmean += p.z;

              std::vector<double> pcoord;
              pcoord.push_back(p.x);
              pcoord.push_back(p.y);
              pcoord.push_back(p.z);
              vertexPoints.push_back(pcoord);
              numPoints++;
          }
          vertexIndices.pop_back(); 
        }
        if(numPoints < 3) continue;

        xmean /= numPoints;
        ymean /= numPoints;
        zmean /= numPoints;

        vertex_t v = add_vertex(graph);
        graph[v].treeId = -1;
        graph[v].points = vertexPoints;
        graph[v].coords[0] = xmean;
        graph[v].coords[1] = ymean;
        graph[v].coords[2] = zmean;

        //vertices.push_back(add_vertex(TreeVertex{instanceId, centroid}, graph));
        vertexPoints.clear();
        vertices.push_back(v);
    }
  }
  return vertices;
}

double euclideanDist(double* vecA, double* vecB){
  return std::sqrt(std::pow(vecA[0]-vecB[0],2) +
            std::pow(vecA[1]-vecB[1],2) +
            std::pow(vecA[2]-vecB[2],2));
}

//Greedy matcher
void computeEdgesGreedy(size_t rowIdx, const std::vector<vertex_t> beamAvertices,
    const std::vector<vertex_t> beamBvertices, graph_t& graph, float distThreshold){

  float dist = 0;
  //std::vector<float> distances;
  int tIdx = 0;
  for(auto vtxA : beamAvertices){
    // if vtxA is the first beam, labels are still -1
    // so we initialize treeId first
    if(rowIdx == (size_t)14){
        graph[vtxA].treeId = tIdx;
        tIdx++;
    }
    for(auto vtxB : beamBvertices){
      dist = euclideanDist(graph[vtxA].coords, graph[vtxB].coords);
      //distances.push_back(dist);
      if(dist < distThreshold && graph[vtxA].treeId != -1){
        graph[vtxB].treeId = graph[vtxA].treeId;
        add_edge(vtxA, vtxB, dist, graph);
      }
    }
  }
}

void computeGraph(const cv::Mat rangeImg, const cv::Mat mask,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, graph_t& graph){
  // mask will always be fixed size but cloud might be smaller/bigger
  size_t maxColIdx = std::min(mask.cols, (int)cloud->width);
  // bottom up
  std::vector<vertex_t> verticesA;
  std::vector<vertex_t> verticesB;
  for(int i=cloud->height-1; i >= 0; --i){
    const unsigned char* range_p = rangeImg.ptr<unsigned char>(i);
    const unsigned char* mask_p = mask.ptr<unsigned char>(i);
    if(i == cloud->height-1){
      verticesA = getVerticesFromBeam(i, maxColIdx, range_p, mask_p, cloud, graph); 
    } else {
      verticesB = getVerticesFromBeam(i, maxColIdx, range_p, mask_p, cloud, graph);
      computeEdgesGreedy(i, verticesA, verticesB, graph, 0.5);
      verticesA = verticesB;
    }
  }
}
/*
int main(int argc, char* argv){
  cv::Mat rangeImg = cv::imread("input.jpg", 0);
  cv::Mat mask = cv::imread("erfnetOut.jpg", 0);
  pcl::PointCloud<PointXYZI>::Ptr cloud (new PointCloud<PointXYZI>);
  if (pcl::io::loadPCDFile<PointXYZI>("cloud.pcd", *cloud) == -1){
     PCL_ERROR("Couldn't read file\n");
  }
  graph_t graph;
  computeGraph(rangeImg, mask, cloud, graph);
}*/
