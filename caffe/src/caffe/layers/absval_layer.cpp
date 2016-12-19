#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "os.h"
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
void AbsValLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

  int numaxis = bottom[0]->num_axes();
  if (numaxis >= 2)
       channels_ = bottom[0]->shape(1);
  else
       channels_ = 1;
  if(numaxis >= 3)
       dim_width = bottom[0]->shape(2);
  else
       dim_width = 1;
  if(numaxis >= 4)
       dim_height = bottom[0]->shape(3);
  else
       dim_height = 1;
  progHandle = glCreateProgram();
  GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(cs,2,csSrc,NULL);
  glCompileShader(cs);
  int rvalue;
  glGetShaderiv(cs,GL_COMPILE_STATUS,&rvalue);
  if (!rvalue) {
      YXLog("rvalue is %d",rvalue);
      YXLog("compile weight program failure");
      fprintf(stderr,"Error in compiling the compute shader\n");
      GLchar log[10240];
      GLsizei length;
      glGetShaderInfoLog(cs,10239,&length,log);
      YXLog("length is %d",length);
      YXLog("Compiler log:/n%s",log);
      fprintf(stderr,"Compiler log:/n%s\n",log);
  }
  glAttachShader(progHandle, cs);
  
  glLinkProgram(progHandle);
  glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
  if (!rvalue) {
      YXLog("link weight program failure");
      fprintf(stderr, "Error in linking compute shader program\n");
      GLchar log[10240];
      GLsizei length;
      glGetProgramInfoLog(progHandle, 10239, &length, log);
      fprintf(stderr, "Linker log:/n%s\n",log);
  }
}

template <typename Dtype>
void AbsValLayer<Dtype>::forward_abs(const GLuint input, const GLuint output) {
       glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,input);
       glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,output);
       glUseProgram(progHandle);
       glDispatchCompute(dim_width, dim_height, channels_);
}

template <typename Dtype>
void AbsValLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
//  caffe_abs(count, bottom[0]->cpu_data(), top_data);
//if(strcmp(this->layer_param_.name().c_str(),"ActivationAbs1")==0)
//  YXLog("mylog6 %.4f %.4f %.4f",top_data[0],top_data[2000],top_data[count-1]);
//if(strcmp(this->layer_param_.name().c_str(),"ActivationAbs1")==0){
//             glBindBuffer(GL_SHADER_STORAGE_BUFFER,bottom[0]->shader_handle());
//             glBufferData(GL_SHADER_STORAGE_BUFFER,count*sizeof(Dtype),bottom_data,GL_DYNAMIC_DRAW);
  forward_abs(bottom[0]->shader_handle(),top[0]->shader_handle());
//             glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[0]->shader_handle());
//             GLint bufMask = GL_MAP_READ_BIT;
//             Dtype* points = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,count*sizeof(Dtype),bufMask);
//             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//}
             double t = cv::getTickCount();
             glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[0]->shader_handle());
             GLint bufMask = GL_MAP_READ_BIT;
             Dtype* points = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,212*sizeof(Dtype),bufMask);
             t = (cv::getTickCount() - t)/cv::getTickFrequency()*1000.;
             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
             YXLog("mylog1777 %s %.2f",this->layer_param_.name().c_str(),t);
}

template <typename Dtype>
void AbsValLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_cpu_sign(count, bottom_data, bottom_diff);
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(AbsValLayer);
#endif

INSTANTIATE_CLASS(AbsValLayer);
REGISTER_LAYER_CLASS(AbsVal);

}  // namespace caffe
