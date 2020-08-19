/**************************************************************************
	Copyright: DANZLE Motion Corporation
	Author: Qiu Guangyue
	Date:2020-08-18
	Description：
        Contains the statement of multi human parsing functions
        (specificlly for torso parsing)
**************************************************************************/
#ifndef MHP_H_
#define MHP_H_

#include <caffe2/core/common.h>
#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>
#include <caffe2/utils/proto_utils.h>
#include <torch/script.h> 

#include <opencv2/opencv.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>


/**************************************************************************
	Author: Qiu Guangyue
	Date: 2020-08-18
    Struct Name: Mask_RCNN
	Struct Description： 
        A custom struct to store img info and infer results
**************************************************************************/
struct img_item
{   
    /***
     * code
     * 0 Succeed
     * 1 Empty input
     * 2 Image height or width is not divisibility of 32
     * 3 Region threshold is bigger than image height
     * 4 No eligible objects in mask rcnn
     * 
    ***/
    int code;
    int person_idx;
    int img_height;
    int img_width;
    float center[2];
    int person_num;
    std::vector<std::vector<int>> person_bbox;
    std::vector<std::vector<float>> real_person_bbox;
    std::vector<float> person_bbox_score;
    cv::Mat instance_masks;
    std::vector<cv::Mat> parsing_results;
};


/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: gl_parsing()
    Function Description： 
        Statement of gl_parsing.
        Human parsing for the global image and the local instance bboxes images. 
    inputs: 
        img_item, cv::Mat, torch::jit::script::Module: SCHP libtorch model
    outputs: 
        img_item
        Added parsing results of the global image and the local instance bboxes images
**************************************************************************/
img_item gl_parsing(img_item item, cv::Mat &inp, torch::jit::script::Module module);


/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: parsing_fusion()
    Function Description： 
        Statement of pasing_fusion().
        Fuse global parsing result and local parsing result. 
        Merge the parsing and instance segmentation information.
    inputs: img_item
    outputs: 
        cv::Mat
        Contains 0~person_idx value to indicate the location of torso for every instance
**************************************************************************/
cv::Mat parsing_fusion(img_item item);


/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: res_pos()
    Function Description： 
        Statement of res_pos().
        Get the position x of every instance's torso from left to right.
    inputs: cv::Mat, int
    outputs: 
        std::map<int,int>
**************************************************************************/
std::map<int,int> res_pos(cv::Mat result_frame, int person_idx);
/**************************************************************************
	Author: Qiu Guangyue
	Date: 2020-08-18
    Class Name: Mask_RCNN
	Class Description： 
        Load Mask_RCNN model. 
        Run Mask_RCNN inference.
**************************************************************************/
class Mask_RCNN
{
private:

    caffe2::NetDef initNet_;
    caffe2::NetDef predictNet_;
    caffe2::Workspace workSpace;

public:
    Mask_RCNN() {}
    virtual ~Mask_RCNN() {}

    /**************************************************************************
        Author: Qiu Guangyue
        Date: 2020-08-18
        Function Name: Mask_RCNN::load_net()
        Function Description： 
            Statement of Loading Mask_RCNN caffe2 model. 
        inputs: std::string, std::string
        outputs: int
    **************************************************************************/
    int load_net(std::string init_model_file, std::string predict_model_file);

    /**************************************************************************
        Author: Qiu Guangyue
        Date: 2020-08-18
        Function Name: Mask_RCNN::run()
        Function Description： 
            Statement of running Mask_RCNN inference and processing result. 
        inputs: cv::Mat, float threshold
        outputs: img_item
    **************************************************************************/
    img_item run(cv::Mat &inp, float threshold=0.0);
};


#endif