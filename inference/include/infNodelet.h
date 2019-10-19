#pragma once

#include <nodelet/nodelet.h>
#include <infNode.h>

namespace trt_inference {
    class InferenceNodelet : public nodelet::Nodelet {
        public:
            InferenceNodelet() {}
            ~InferenceNodelet() {}
            virtual void onInit();
        private:
            Inference::Ptr inference;
    };
}
