#pragma once

#include <nodelet/nodelet.h>
#include <infNodeCPU.h>

namespace trt_inference {
    class InferenceNodeletCPU : public nodelet::Nodelet {
        public:
            InferenceNodeletCPU() {}
            ~InferenceNodeletCPU() {}
            virtual void onInit();
        private:
            Inference::Ptr inference;
    };
}
