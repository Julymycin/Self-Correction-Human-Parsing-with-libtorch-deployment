#include <caffe2/core/common.h>
#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/opencv.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>

using namespace std;
using namespace caffe2;
using namespace cv;


struct ins_pre{
    caffe2::Tensor pred_boxes; // x1,y1,x2,y2
    caffe2::Tensor scores;
    caffe2::Tensor pred_classes;
    caffe2::Tensor pred_masks;
    int image_size[2]; //[h,w]
    std::string img_path; //full path
}

struct img_item{
    std::string img_path;
    std::string img_name;
    int img_height;
    int img_width;
    float* center;
    int preson_num;
    std::vector<std::vector<int>> person_bbox;
    std::vector<std::vector<float>> real_person_bbox;
    std::vector<float> person_bbox_score;
    cv::Mat instance_masks;
    std::vector<cv::Mat> parsing_results;
}

class Mask_RCNN
{
	private:
		double threshold_;
//		std::vector<Box> mask_rcnn_box;
		caffe2::NetDef initNet_;
		caffe2::NetDef predictNet_;
		Workspace workSpace;

	public:
		Mask_RCNN()  {}
		virtual ~Mask_RCNN() {}
		void load_net(std::string init_model_file, std::string predict_model_file);

		ins_pre run(cv::Mat& input);
};