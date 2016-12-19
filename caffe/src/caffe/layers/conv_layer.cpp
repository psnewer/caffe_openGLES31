#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
//      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
//          top_data + n * this->top_dim_);
//if(strcmp(this->layer_param_.name().c_str(),"Conv1")==0)
//      YXLog("mylog6 %.4f %.4f %.4f",top_data[15],top_data[2000],top_data[this->top_dim_-1]);

//        if(strcmp(this->layer_param_.name().c_str(),"Conv1")==0){
//             glBindBuffer(GL_SHADER_STORAGE_BUFFER,bottom[i]->shader_handle());
//             glBufferData(GL_SHADER_STORAGE_BUFFER,this->bottom_dim_*sizeof(Dtype),bottom_data + n * this->bottom_dim_,GL_DYNAMIC_DRAW);
             this->forward_shader_gemm(bottom[i]->shader_handle(),this->blobs_[0]->shader_handle(),top[i]->shader_handle(),caffe_getaxillary_storage());
//             glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[i]->shader_handle());
//             GLint bufMask = GL_MAP_READ_BIT;
//             Dtype* points = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,this->top_dim_*sizeof(Dtype),bufMask);
//             memcpy(top_data + n * this->top_dim_,points,this->top_dim_*sizeof(Dtype));
//             YXLog("mylog6 %.4f %.4f %.4f",points[15],points[2000],points[this->top_dim_-1]);
//             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//       }
             double t = cv::getTickCount();
             glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[0]->shader_handle());
             GLint bufMask = GL_MAP_READ_BIT;
             Dtype* points = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,212*sizeof(Dtype),bufMask);
             t = (cv::getTickCount() - t)/cv::getTickFrequency()*1000.;
             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
             YXLog("mylog17771 %s %.2f",this->layer_param_.name().c_str(),t);

      if (this->bias_term_) {
//        const Dtype* bias = this->blobs_[1]->cpu_data();
//       this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
//if(strcmp(this->layer_param_.name().c_str(),"Conv1")==0)
//        YXLog("mylog6 %.4f %.4f %.4f",top_data[0],top_data[2000],top_data[this->top_dim_-1]);
//if(strcmp(this->layer_param_.name().c_str(),"Conv1")==0){
this->forward_shader_bias(this->blobs_[1]->shader_handle(),top[i]->shader_handle());
//if(strcmp(this->layer_param_.name().c_str(),"Conv4")==0){
 //            glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[i]->shader_handle());
 //            GLint bufMask = GL_MAP_READ_BIT;
//             Dtype* points = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,this->top_dim_*sizeof(Dtype),bufMask);
//             memcpy(top_data + n * this->top_dim_,points,this->top_dim_*sizeof(Dtype));
//YXLog("mylog6 %s %.2f,%.2f,%.2f",this->layer_param_.name().c_str(),points[0],points[100],points[this->top_dim_-1]);
//             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//}
//}
             double t = cv::getTickCount();
             glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[0]->shader_handle());
             GLint bufMask = GL_MAP_READ_BIT;
             Dtype* points = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,212*sizeof(Dtype),bufMask);
             t = (cv::getTickCount() - t)/cv::getTickFrequency()*1000.;
             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
             YXLog("mylog17772 %s %.2f",this->layer_param_.name().c_str(),t);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_shader(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const GLuint weight= this->blobs_[0]->shader_handle();
  for (int i = 0; i < bottom.size(); ++i) {
    const GLuint bottom_data = bottom[i]->shader_handle();
    const GLuint top_data = top[i]->shader_handle();
    for (int n = 0; n < this->num_; ++n) {
//      this->forward_shader_gemm(bottom_data, weight,
//          top_data);
      if (this->bias_term_) {
        const GLuint bias = this->blobs_[1]->shader_handle();
        this->forward_shader_bias(top_data, bias);
      }
    }
  }
             double t = cv::getTickCount();
             glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[0]->shader_handle());
             GLint bufMask = GL_MAP_READ_BIT;
             Dtype* points = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,212*sizeof(Dtype),bufMask);
             t = (cv::getTickCount() - t)/cv::getTickFrequency()*1000.;
             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
             YXLog("mylog1777 %s %.2f",this->layer_param_.name().c_str(),t);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
