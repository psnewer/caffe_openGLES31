#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/opencv.hpp>
#include <os.h>

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  M_ = bottom[0]->count(0, axis);
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  int numaxis = bottom[0]->num_axes();  
  int in_width = 1;
  int in_height = 1;
  int conv_in_channels = bottom[0]->shape(axis);
  if (numaxis>axis+1)
      in_width = bottom[0]->shape(axis+1);
  if (numaxis>axis+2)
      in_height = bottom[0]->shape(axis+2);

  unsigned int channels = 0;
  if (conv_in_channels*in_width*in_height > 1024)
     channels = 2;
  else
     channels =1;
  csSrc[0] = _csSrc[0];
  csSrc[1] = new char[5000];
  sprintf(csSrc[1],_csSrc[1],conv_in_channels/channels,in_width,in_height);

  progHandle = glCreateProgram();
  YXLog("the progHandle value is %d",progHandle);
  GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
  YXLog("the shaderhandle value is %d",cs);
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

  rvalue = 0;
  biasHandle = glCreateProgram();
  GLuint bias = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(bias,2,biasSrc,NULL);
  glCompileShader(bias);
  glGetShaderiv(bias,GL_COMPILE_STATUS,&rvalue);
  if (!rvalue) {
      YXLog("compile bias program failure");
      fprintf(stderr,"Error in compiling the compute shader\n");
      GLchar log[10240];
      GLsizei length;
      glGetShaderInfoLog(bias,10239,&length,log);
      YXLog("Bias Compiler log:/n%s",log);
      fprintf(stderr,"Compiler log:/n%s\n",log);
  }
  glAttachShader(biasHandle, bias);
  
  glLinkProgram(biasHandle);
  glGetProgramiv(biasHandle, GL_LINK_STATUS, &rvalue);
  if (!rvalue) {
      YXLog("link bias program failure");
      fprintf(stderr, "Error in linking compute shader program\n");
      GLchar log[10240];
      GLsizei length;
      glGetProgramInfoLog(progHandle, 10239, &length, log);
      fprintf(stderr, "Linker log:/n%s\n",log);
  }

  YXLog("begin bind uniform variables");
  glUseProgram(progHandle);
  glUniform1ui(glGetUniformLocation(progHandle, "cellsize"), conv_in_channels*in_width*in_height/channels);
  glUniform1ui(glGetUniformLocation(progHandle, "channel"), channels);
  YXLog("bind uniform variables done");
  delete csSrc[1];
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    caffe_shader_set(bias_multiplier_.count(), 0, Dtype(1),
        bias_multiplier_.shader_handle());
  }

}

template <typename Dtype>
void InnerProductLayer<Dtype>::forward_shader_gemm(const GLuint input,
    const GLuint weights, const GLuint output, const GLuint tmpbuffer) {
       glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,input);
       glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,weights);
       glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,output);
       glBindBufferBase(GL_SHADER_STORAGE_BUFFER,20,tmpbuffer);
       glUseProgram(progHandle);
       glDispatchCompute(M_, N_, 1);
       glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::forward_shader_bias(const GLuint bias,
    const GLuint output) {
       glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,bias);
       glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,bias_multiplier_.shader_handle());
       glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,output);
       glUseProgram(biasHandle);
       glDispatchCompute(M_, N_, 1);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

//  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
//      M_, N_, K_, (Dtype)1.,
//      bottom_data, weight, (Dtype)0., top_data);
//if(strcmp(this->layer_param_.name().c_str(),"Dense1")==0)
//YXLog("mylog6 %.4f %.4f %.4f",top_data[0],top_data[100],top_data[127]);

//        if(strcmp(this->layer_param_.name().c_str(),"Dense1")==0){
//             glBindBuffer(GL_SHADER_STORAGE_BUFFER,bottom[0]->shader_handle());
//             glBufferData(GL_SHADER_STORAGE_BUFFER,K_*sizeof(Dtype),bottom_data,GL_DYNAMIC_DRAW);
//             glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[0]->shader_handle());
//             GLint bufMask = GL_MAP_READ_BIT;
//             Dtype* points_ = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,M_*N_*sizeof(Dtype),bufMask);
//             memset(points_,0,M_*N_*sizeof(Dtype));
//             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
             this->forward_shader_gemm(bottom[0]->shader_handle(),this->blobs_[0]->shader_handle(),top[0]->shader_handle(), caffe_getaxillary_storage());
//if(strcmp(this->layer_param_.name().c_str(),"Dense1")==0){
//             glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[0]->shader_handle());
//             GLint bufMask = GL_MAP_READ_BIT;
//             Dtype* points = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,M_*N_*sizeof(Dtype),bufMask);
//            YXLog("mylog6 %.4f %.4f %.4f",points[0],points[100],points[127]);
//             memcpy(top_data ,points,128*sizeof(Dtype));
//             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//}
//             glBindBuffer(GL_SHADER_STORAGE_BUFFER,top[0]->shader_handle());
//             Dtype* points = (Dtype*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,M_*N_*sizeof(Dtype),bufMask);
//             memcpy(top_data,points,M_*N_*sizeof(Dtype));
////             YXLog("mylog6 %.4f %.4f %.4f",points[0],points[100],points[127]);
//             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//       }

  if (bias_term_) {
//    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
//        bias_multiplier_.cpu_data(),
//        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
//if(strcmp(this->layer_param_.name().c_str(),"Dense2")==0)
//YXLog("mylog6 %.4f %.4f %.4f",top_data[0],top_data[100],top_data[127]);
//if(strcmp(this->layer_param_.name().c_str(),"Dense2")==0){
this->forward_shader_bias(this->blobs_[1]->shader_handle(),top[0]->shader_handle());

//if(strcmp(this->layer_param_.name().c_str(),"Dense2")==0){
//            YXLog("mylog6 %.4f %.4f %.4f",points[0],points[100],points[211]);
//             memcpy(top_data ,points,212*sizeof(Dtype));
//             for(int i = 0;i<128;i++)
//                top_data[i]=points[i];
//             glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//}
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
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
