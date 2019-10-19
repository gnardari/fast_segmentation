#pragma once

#include <nodelet/nodelet.h>
#include <segNodeCPU.h>

namespace trt_inference {
    class SegmentationNodelet : public nodelet::Nodelet {
        public:
            SegmentationNodelet() {}
            ~SegmentationNodelet() {}
            virtual void onInit();
        private:
            Segmentation::Ptr segmentation;
    };
}
