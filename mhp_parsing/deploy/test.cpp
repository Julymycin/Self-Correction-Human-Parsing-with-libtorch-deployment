#include "mhp.h"
#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>

#include <caffe2/core/blob.h>
#include <caffe2/core/common.h>
#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>
#include <caffe2/utils/proto_utils.h>

using namespace std;
using namespace caffe2;
using namespace cv;

long long get_timestamp()
{
        auto tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
        auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
        const long long timestamp = tmp.count();
        return timestamp;
}
int main()
{
    long long start,end;
    
    // std::string img_path="/home/qiu/Projects/Self-Correction-Human-Parsing/mhp_parsing/data/test/src_imgs/1.jpg";
    std::string img_path="/home/qiu/Projects/Self-Correction-Human-Parsing/run/garmin-forerunner-245-what-to-expect.jpg";
    std::string mm_init_pb="/home/qiu/Projects/detectron2-master/tools/deploy/caffe2_model_cihp_person/model_init.pb";
    std::string mm_pb="/home/qiu/Projects/detectron2-master/tools/deploy/caffe2_model_cihp_person/model.pb";
    std::string ckpt_path = "/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/torso_pascal_abn.pt";
    cv::Mat input=cv::imread(img_path);
    cv::Mat split_stitch_frame=input.clone();
    // cvtColor(input, input, CV_BGR2GRAY);
    // cvtColor(input, input, CV_GRAY2BGR);
    // cv::resize(input,input,cv::Size(input.cols/32*32,input.rows/32*32));
    // cv::Mat hp_bg_frame = cv::Mat::zeros(cv::Size(split_stitch_frame.cols / 32 * 32 + 32, split_stitch_frame.rows / 32 * 32 + 32), CV_8UC3);
    // cv::resize(split_stitch_frame, hp_bg_frame(cv::Rect(0,0,split_stitch_frame.cols,split_stitch_frame.rows)), split_stitch_frame.size());
    cv::Mat hp_bg_frame;
    cv::resize(split_stitch_frame,hp_bg_frame,cv::Size(split_stitch_frame.cols / 32 * 32,split_stitch_frame.rows / 32 * 32));
    Mask_RCNN mm;
    mm.load_net(mm_init_pb,mm_pb);
    torch::jit::script::Module pascal_module;
    pascal_module = torch::jit::load(ckpt_path);

    start=get_timestamp();
    img_item item=mm.run(hp_bg_frame);
    std::cout<<"-------run end------"<<endl;
    cv::Mat parsing=parsing_fusion(gl_parsing(item,hp_bg_frame,pascal_module));
    std::map<int,int> pos=res_pos(parsing, item.person_idx);
    std::cout<<pos<<endl;
    std::cout<<"item code: "<<item.code<<endl;
    parsing=parsing*255/5;
    end=get_timestamp();
    cout<<"---"<<(end-start)/1.0<<endl;
    cv::imshow("ss",parsing);
    cv::waitKey(-1);
    cout<<"----------end--------"<<endl;
    return 0;
}