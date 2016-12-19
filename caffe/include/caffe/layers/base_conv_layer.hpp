#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_shader_gemm(const GLuint input, const GLuint weights,
      const GLuint output, const GLuint tmpbuffer);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void forward_shader_bias(const GLuint output, const GLuint bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int out_spatial_dim_width;
  int out_spatial_dim_height;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

  GLuint progHandle;
  GLuint biasHandle;

  char* csSrc[2];
  char *_csSrc[2] = {
        "#version 310 es\n\
         #extension GL_ARB_gpu_shader_fp64: enable\n",
        "uniform int in_width;\
         uniform int in_height;\
         uniform int padding;\
         uniform int stride;\
         uniform uint cellsize;\
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
             	int x = int(gl_WorkGroupID.x)*stride - padding;\
                int y = int(gl_WorkGroupID.y)*stride - padding;\
                int current_x = x + int(gl_LocalInvocationID.x);\
		int current_y = y + int(gl_LocalInvocationID.y);\
                if (current_x>=0&&current_x<in_width&&current_y>=0&&current_y<in_height)\
	            {\
                     tmp[(gl_WorkGroupID.z*gl_NumWorkGroups.y*gl_NumWorkGroups.x+gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x)*cellsize+gl_LocalInvocationIndex]=bottom[in_width*in_height*int(gl_LocalInvocationID.z) + current_y*in_width + current_x] * weights[gl_WorkGroupID.z*cellsize + gl_LocalInvocationID.z*gl_WorkGroupSize.x*gl_WorkGroupSize.y + gl_LocalInvocationID.y*gl_WorkGroupSize.x + gl_LocalInvocationID.x];\
                     }\
                else\
                    {\
                     tmp[(gl_WorkGroupID.z*gl_NumWorkGroups.y*gl_NumWorkGroups.x+gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x)*cellsize+gl_LocalInvocationIndex]=0.0;\
                    }\
                groupMemoryBarrier();\
                if (int(gl_LocalInvocationIndex)==0)\
                    {\
                     top[gl_WorkGroupID.z*gl_NumWorkGroups.y*gl_NumWorkGroups.x+gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x]=0.0;\
                     for (uint i=uint(0);i<cellsize;i++)\
	                 top[gl_WorkGroupID.z*gl_NumWorkGroups.y*gl_NumWorkGroups.x+gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x] += tmp[(gl_WorkGroupID.z*gl_NumWorkGroups.y*gl_NumWorkGroups.x+gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x)*cellsize+i];\
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
	top[gl_WorkGroupID.z*gl_NumWorkGroups.y*gl_NumWorkGroups.x + gl_WorkGroupID.y*gl_NumWorkGroups.x + gl_WorkGroupID.x] += bottom[gl_WorkGroupID.z] * multiplier[gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x];\
     }"
  };

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int kernel_width;
  int kernel_height;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
