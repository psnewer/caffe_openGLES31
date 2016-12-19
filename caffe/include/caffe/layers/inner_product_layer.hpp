#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_shader(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void forward_shader_gemm(const GLuint input, const GLuint weights,
      const GLuint output, const GLuint tmpbuffer);
  void forward_shader_bias(const GLuint output, const GLuint bias);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

  GLuint progHandle;
  GLuint biasHandle;

  char *csSrc[2];
  char *_csSrc[2] = {
        "#version 310 es\n",
        "uniform uint cellsize;\
         uniform uint channel;\
         layout(std430,binding=1) buffer input_\
           {\
             float bottom[];\
           };\
         layout(std430,binding=2) buffer weights_\
           {\
             float weights[];\
           };\
         layout(std430,binding=3) buffer output_\
           {\
             float top[];\
           };\
         layout(std430,binding=20) buffer tmp_\
           {\
             float tmp[];\
           };\
         layout (local_size_x = %u, local_size_y = %u, local_size_z = %u) in;\
         void main() {\
                uint weightInd = channel*cellsize*gl_WorkGroupID.y+gl_LocalInvocationIndex;\
                uint dataInd = channel*cellsize*gl_WorkGroupID.x+gl_LocalInvocationIndex;\
                tmp[cellsize*gl_WorkGroupID.y+gl_LocalInvocationIndex] = weights[weightInd]*bottom[dataInd];\
if(channel == uint(2))\
{\
                weightInd += cellsize;\
                dataInd += cellsize;\
                tmp[cellsize*gl_WorkGroupID.y+gl_LocalInvocationIndex] += weights[weightInd]*bottom[dataInd];\
}\
                groupMemoryBarrier();\
                if (int(gl_LocalInvocationIndex)==0)\
                    {\
                      top[gl_WorkGroupID.x*gl_NumWorkGroups.y+gl_WorkGroupID.y] = 0.0;\
                      for (uint i=uint(0);i<cellsize;i++)\
	                  top[gl_WorkGroupID.x*gl_NumWorkGroups.y+gl_WorkGroupID.y] += tmp[cellsize*gl_WorkGroupID.y+i];\
                }\
          }"
    };

  const char *biasSrc[2] = {
    "#version 310 es\n\
     #extension GL_ARB_gpu_shader_fp64: enable\n",
    "layout(std430,binding=1) buffer input_\
     {\
       float bottom[];\
     };\
     layout(std430,binding=2) buffer multiplier_\
     {\
       float multiplier[];\
     };\
     layout(std430,binding=3) buffer output_\
     {\
       float top[];\
     };\
     layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\
     void main() {\
	top[gl_WorkGroupID.x*gl_WorkGroupSize.y+gl_WorkGroupID.y] += multiplier[gl_WorkGroupID.x]*bottom[gl_WorkGroupID.y];\
     }"
  };
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
