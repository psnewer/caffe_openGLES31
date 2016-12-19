#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }

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

  void forward_pool(const GLuint input, const GLuint output);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;

  GLuint progHandle;

  const char *csSrc[2] = {
        "#version 310 es\n\
         #extension GL_ARB_gpu_shader_fp64: enable\n",
        "uniform int in_width;\
         uniform int in_height;\
         uniform int kernel_width;\
         uniform int kernel_height;\
         uniform int padding;\
         uniform int stride;\
         layout(std430,binding=1) buffer input_\
           {\
             float bottom[];\
           };\
         layout(std430,binding=2) buffer output_\
           {\
             float top[];\
           };\
         layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\
         void main() {\
             	int x_start = int(gl_WorkGroupID.x)*stride - padding;\
                int y_start = int(gl_WorkGroupID.y)*stride - padding;\
                int x_end = min(x_start + kernel_width,in_width);\
                int y_end = min(y_start + kernel_height,in_height);\
                x_start = max(0, x_start);\
                y_start = max(0, y_start);\
                float val = bottom[int(gl_WorkGroupID.z)*in_width*in_height + y_start * in_width + x_start];\
                for(int i = x_start; i < x_end; i++)\
                   for(int j = y_start; j < y_end; j++)\
                       {\
                           val = max(val,bottom[int(gl_WorkGroupID.z)*in_width*in_height + j * in_width + i]);\
                       }\
	        top[gl_WorkGroupID.z*gl_NumWorkGroups.y*gl_NumWorkGroups.x + gl_WorkGroupID.y*gl_NumWorkGroups.x + gl_WorkGroupID.x] = val;\
         }"
    };
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
