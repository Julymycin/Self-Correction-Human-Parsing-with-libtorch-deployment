/**************************************************************************
	Copyright: DANZLE Motion Corporation
	Author: Qiu Guangyue
	Date:2020-08-18
	Description：
        Contains the difinition of multi human parsing functions
        (specificlly for torso parsing)
**************************************************************************/

#include "mhp.h"
#include <torch/script.h> 
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

#define PI 3.141592653589793238462

// ------------------------------------------------------------------------------
/* 
Following are 4 util functions, for warpAffine
Shouldn't be modified
Including get_3rd_point, get_dir, get_affine_transform, parsing_transform
*/

/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: get_3rd_point()
    Function Description： 
        Three points is needed for warpAffine, use two to get the third one
    inputs: cv::Point2f, cv::Point2f
    outputs: cv::Point2f
**************************************************************************/
cv::Point2f get_3rd_point(cv::Point2f a, cv::Point2f b)
{
    cv::Point2f direct = a - b;
    cv::Point2f res;
    res.x = b.x + direct.y;
    res.y = b.y - direct.x;
    return res;
}

/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: get_dir()
    Function Description： 
        When rotation is not None, get the direction
    inputs: cv::Point2f, float
    outputs: cv::Point2f
**************************************************************************/
cv::Point2f get_dir(cv::Point2f src_point, float rot_rad)
{
    float sn = sin(rot_rad);
    float cs = cos(rot_rad);

    cv ::Point2f src_result(0, 0);
    src_result.y = src_point.y * cs - src_point.x * sn;
    src_result.x = src_point.y * sn + src_point.x * cs;

    return src_result;
}

/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: get_affine_transform()
    Function Description： 
        Get the transform Mat of warpAffine
    inputs: cv::Point2f, float, float, cv::Size, int
    outputs: cv::Mat
**************************************************************************/
cv::Mat get_affine_transform(cv::Point2f center, float scale, float rot, cv::Size output_size, int inv=0)
{
    int src_w = scale;
    int dst_w = output_size.width;
    int dst_h = output_size.height;
    float rot_rad = PI * rot / 180;
    cv::Point2f src_dir = get_dir(cv::Point2f(src_w * -0.5, 0), rot_rad);

    cv::Point2f dst_dir((dst_w - 1) * -0.5, 0);
    cv::Point2f src[3];
    cv::Point2f dst[3];
    src[0] = center;
    src[1] = center + src_dir;
    dst[0] = cv::Point2f((dst_w - 1) * 0.5, (dst_h - 1) * 0.5);
    dst[1] = cv::Point2f((dst_w - 1) * 0.5, (dst_h - 1) * 0.5) + dst_dir;
    src[2] = get_3rd_point(src[0], src[1]);
    dst[2] = get_3rd_point(dst[0], dst[1]);
    cv::Mat trans;

    // use inv to choose source to dst or contrary
    if(inv){
        trans = cv::getAffineTransform(dst, src);
    }
    else{
        trans = cv::getAffineTransform(src, dst);
    }
    return trans;
}

/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: parsing_transform()
    Function Description： 
        Transform parsing result to orignal size
    inputs: cv::Mat, cv::Mat, cv::Point2f, float, cv::Size, cv::Size
    outputs: cv::Mat
**************************************************************************/
cv::Mat parsing_transform(cv::Mat parsing,  cv::Mat transform_parsing, cv::Point2f center, float scale, cv::Size img_size,cv::Size input_size){

    cv::Mat trans=get_affine_transform(center,scale, 0, input_size,1);

    cv::warpAffine(parsing,transform_parsing,trans,img_size,0);


    return transform_parsing;
}
/* 
Above are 4 util functions for warpAffine
Shouldn't be modified
Including get_3rd_point, get_dir, get_affine_transform, parsing_transform
*/
// ------------------------------------------------------------------------------


/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: Mask_RCNN::load_net()
    Function Description： 
        Definition of Loading caffe2 Mask_RCNN model. 
    inputs: 
        std::string init_model_file: *_init.pb
        std::string predict_model_file: *.pb
    outputs: int
**************************************************************************/
int Mask_RCNN::load_net(std::string init_model_file, std::string predict_model_file)
{
    CAFFE_ENFORCE(ReadProtoFromFile(init_model_file, &initNet_));
    CAFFE_ENFORCE(ReadProtoFromFile(predict_model_file, &predictNet_));
    CAFFE_ENFORCE(workSpace.RunNetOnce(initNet_));
    return 0;
}


/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: Mask_RCNN::run()
    Function Description： 
        Definition of running Mask_RCNN inference and processing result.
        Run inference of mask rcnn, parse the result to get bboxes, labels, scores, masks.
        Filter the result by score threshold and labels.
        Merge masks(28*28*num_instance) to a panoptic segmentation mask(H*W*1).
        Expand bboxes to 1.5 times bigger.
    inputs: 
        cv::Mat &inp: the split image
    outputs: 
        img_item 
        Includeing img's height, width, center, number of person, bboxes, 
            expanded bboxes, confidence score and panoptic segmentation mask.
**************************************************************************/
img_item Mask_RCNN::run(cv::Mat &inp)
{

    cv::Mat input=inp.clone();
    int height = input.rows;
    int width = input.cols;
    img_item item; // img_item struct to store the inference result

    // FPN models require divisibility of 32
    assert(height % 32 == 0 && width % 32 == 0);
    // mask rcnn model requires 1 batch and 3 channel to infer
    const int batch = 1;
    const int channels = 3;

    // initialize Net and Workspace
    for (auto& str : predictNet_.external_input()) {
        workSpace.CreateBlob(str);
    }

    // setup inputs to fit the format of mask rcnn model
    auto data = BlobGetMutableTensor(workSpace.GetBlob("data"), caffe2::CPU);
    data->Resize(batch, channels, height, width);
    float* ptr = data->mutable_data<float>();
    // HWC to CHW
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < height * width; ++i) {
        ptr[c * height * width + i] = static_cast<float>(input.data[3 * i + c]);
        }
    }

    auto im_info =
      BlobGetMutableTensor(workSpace.GetBlob("im_info"), caffe2::CPU);
    im_info->Resize(batch, 3);
    float* im_info_ptr = im_info->mutable_data<float>();
    im_info_ptr[0] = height;
    im_info_ptr[1] = width;
    im_info_ptr[2] = 1.0;

    // start inference
    CAFFE_ENFORCE(workSpace.RunNetOnce(predictNet_));

    // parse Mask R-CNN outputs
    caffe2::Tensor bbox(
        workSpace.GetBlob("bbox_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
    caffe2::Tensor scores(
        workSpace.GetBlob("score_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
    caffe2::Tensor labels(
        workSpace.GetBlob("class_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
    caffe2::Tensor mask_probs(
        workSpace.GetBlob("mask_fcn_probs")->Get<caffe2::Tensor>(), caffe2::CPU);

    int img_h = height;
    int img_w = width;
    float exp_ratio = 1.5; // expanding ratio to expand the instnce bboxes 
    // panoptic_seg, the panoptic segmentation mask
    Mat panoptic_seg = Mat::zeros(cv::Size(img_w,img_h), CV_8UC1);
    int num_instances = bbox.sizes()[0];
    std::vector<std::vector<int>> exp_bbox; // expanded bboxes
    std::vector<std::vector<float>> ori_bbox; // origin bboxes
    std::vector<float> bbox_score_list; // bbox confidence score

    int person_idx = 0;
    for (int i = 0; i < num_instances; ++i)
    {
        
        cv::Mat temp_mask = Mat::zeros(cv::Size(img_w,img_h), CV_8UC1);

        float score = scores.data<float>()[i]; // the confidence score of the bbox
        int label = labels.data<float>()[i]; // the classification of the bbox
        // if confidece score is too low, or the class is not 'person', 
        // skip them
        if (score < 0.6 || label != 0)
            continue; 
        const float* box = bbox.data<float>() + i * 4; // bbox
		

        const float* mask = mask_probs.data<float>() +
			i * mask_probs.size_from_dim(1) + label * mask_probs.size_from_dim(2); // mask, 28*28
        // get the location of the bbox in the origin image
        cv::Rect Box_=cv::Rect(box[0],box[1],box[2]-box[0]+1,box[3]-box[1]+1);

        // resize the corresponding mask to bbox's shape, and restore to the temp_mask(H*W*1)
		cv::Mat cv_mask(28, 28, CV_32FC1);
        memcpy(cv_mask.data, mask, 28 * 28 * sizeof(float));
		cv::resize(cv_mask,cv_mask,cv::Size(box[2]-box[0]+1,box[3]-box[1]+1));
		cv::Mat mask_=(cv_mask>0.3);
        cv::Mat mask_roi=temp_mask(Box_);
        mask_roi=mask_roi+mask_;

        // caculate the mask area in temp_mask
        float mask_area = cv::sum(temp_mask>0)[0];
        if (mask_area == 0)
            continue;
        /*
        caculate the intersection area of temp_mask and panoptic_seg
        if the intersection ratio is higher than the threshold
        means that the mask seems to be re-identified, drop it
        */
        Mat intersetc = (temp_mask > 0) & (panoptic_seg > 0);
        float intersect_area = cv::sum(intersetc)[0];
        if (intersect_area * 1.0 / mask_area > 0.5)
            continue;
        if (intersect_area > 0)
            temp_mask = temp_mask & (panoptic_seg == 0);
        person_idx += 1;
        // assign the person_idx to the mask, and merge it to panoptic_seg
        Mat dist1 = temp_mask > 0;
        dist1 = dist1 / 255 * person_idx;
        panoptic_seg = panoptic_seg | dist1;

        bbox_score_list.push_back(score);
        // expand the origin bbox for next-step
        float x_min = box[0];
        float y_min = box[1];
        float x_max = box[2];
        float y_max = box[3];
        float exp_x = (x_max - x_min) * ((exp_ratio - 1) / 2);
        float exp_y = (y_max - y_min) * ((exp_ratio - 1) / 2);
        int new_x_min = max(0, (int)round(x_min - exp_x));
        int new_y_min = max(0, (int)round(y_min - exp_y));
        int new_x_max = min(img_w - 1, (int)round(x_max + exp_x));
        int new_y_max = min(img_h - 1, (int)round(y_max + exp_y));
        int exp_box[4]={new_x_min, new_y_min, new_x_max, new_y_max};
        float ori_box[4]={x_min, y_min, x_max, y_max};
        exp_bbox.push_back(vector<int>(exp_box,exp_box+4));
        ori_bbox.push_back(vector<float>(ori_box,ori_box+4));

    }
    // if no eligible obejects detected, return a img_item with 0 person_idx
    if (person_idx == 0)
    {
        item.person_idx=person_idx;
        return item;
    }
    item.person_idx=person_idx;
    item.img_height = img_h;
    item.img_width = img_w;
    item.center[0] = (img_h-1) / 2.0; 
    item.center[1] = (img_w-1) / 2.0;
    item.person_num = person_idx;
    item.person_bbox = exp_bbox;
    item.real_person_bbox = ori_bbox;
    item.person_bbox_score = bbox_score_list;
    item.instance_masks = panoptic_seg;

    return item;
}

/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: gl_parsing()
    Function Description： 
        Human parsing for the global image and the local instance bboxes images. 
    inputs: 
        img_item, cv::Mat, torch::jit::script::Module: SCHP libtorch model
    outputs: 
        img_item
        Added parsing results of the global image and the local instance bboxes images
**************************************************************************/
img_item gl_parsing(img_item item, cv::Mat &inp, torch::jit::script::Module module)
{   
    // if the person_idx==0 in item, no eligible objects, pass
    if(item.person_idx==0){
        return item;
    }

    cv::Mat input=inp.clone();
    std::vector<cv::Mat> parsing_results;
    cv::Size input_size(512,512);
    // a loop for inference of the global image and the local instance bboxes images
    // i==0, global image
    // i>0, local instance bboxes images
    for (int i = 0; i <= item.person_num; i++)
    {
        cv::Mat img=input.clone();

        // use the expand bbox to get a Rect to crop the image
        if(i!=0)
        {
            int x_min = item.person_bbox[i - 1][0];
            int y_min = item.person_bbox[i - 1][1];
            int temp_w = item.person_bbox[i - 1][2] - x_min+1;
            int temp_h = item.person_bbox[i - 1][3] - y_min+1;
            Rect rect(x_min, y_min, temp_w, temp_h);
            img = input(rect);
        }
        
        int height = img.rows;
        int width = img.cols;
        // instead of resize, warpAffine is required for the pre_process of the inputs of SCHP
        // firstly , get the parameters of warpAffine: center, scale, rotation, origin size, output size 
        // then, transform the img(origin size) to img_temp(output size, alos the input size of SCHP) using warpAffine
        cv::Point2f c((width-1) / 2.0, (height-1) / 2.0); // center of the image
        cv::Size img_size(width,height);
        float s = max(height, width) - 1;
        float r = 0;
        cv::Mat trans=get_affine_transform(c,s,r,input_size);
        cv::Mat img_temp;
        cv::warpAffine(img,img_temp,trans,input_size);

        // convert imgs to libtorch format(float, tenosr, normalize, {-1, c, h, w})
        cv::Mat img_float;
        img_temp.convertTo(img_float, CV_32F, 1.0 / 255);
        
        cv::resize(img_float, img_float, input_size);
        auto img_tensor = torch::from_blob(img_float.data, {1, input_size.width, input_size.height, 3});
        img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
        img_tensor[0][0] = img_tensor[0][0].sub_(0.406).div_(0.225);
        img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
        img_tensor[0][2] = img_tensor[0][2].sub_(0.485).div_(0.229);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img_tensor.to(torch::kCUDA));

        // inference result
        torch::Tensor output = module.forward(inputs).toTensor(); // Tensor [NUM_CLASSES, 128, 128]
        // upsmale result to input_size(512)
        auto res = torch::upsample_bilinear2d(output, {input_size.width, input_size.height}, true); // Tensor [1, NUM_CLASSES, input_s, input_s]
        res = torch::squeeze(res);                                                     // Tensor [NUM_CLASSES, input_s, input_s]
        res = res.permute({1, 2, 0});                                                  //CHW->HWC

        // use torch::max in dim 2 to get the segmentation result
        torch::Tensor resa=std::get<1>(torch::max(res,2)); // Tensor [1, inputsize, inputsize]
        
        // transform tensor to cv::Mat
        resa = resa.to(torch::kCPU).squeeze().detach().to(torch::kU8);
        cv::Mat parsing_result(input_size.height, input_size.width, CV_8UC1);
        std::memcpy((void *)parsing_result.data, resa.data_ptr(), sizeof(torch::kU8) * resa.numel());

        parsing_result=(parsing_result==1); // for torso_pascal

        // use warpAffine to tranform parsing result[1, inputsize, inputsize] to original size
        cv::Mat transform_parsing;
        transform_parsing=parsing_transform(parsing_result,transform_parsing, c,s,img_size,input_size);
        
        // store the result to item
        parsing_results.push_back(transform_parsing);
        
        inputs.pop_back();
    }
    item.parsing_results=parsing_results;
    return item;
}


/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: parsing_fusion()
    Function Description： 
        Fuse global parsing result and local parsing result. 
        Merge the parsing and instance segmentation information
    inputs: img_item
    outputs: 
        cv::Mat
        Contains 0~person_idx value to indicate the location of torso for every instance
**************************************************************************/
cv::Mat parsing_fusion(img_item item){
    // No eligible instance objects
    if(item.person_idx==0){
        return Mat::zeros(cv::Size(1, 1), CV_8UC1);
    }
    int img_height=item.img_height;
    int img_width=item.img_width;
    std::vector<std::vector<int>> msrcnn_bbox=item.person_bbox;
    std::vector<float> bbox_score=item.person_bbox_score;
    std::vector<cv::Mat> parsing_results=item.parsing_results;
    cv::Mat instance_masks=item.instance_masks;

    cv::Mat global_parsing=parsing_results[0];
    Mat global_temp = Mat::zeros(cv::Size(img_width, img_height), CV_8UC1);
    // fuse global parsing result and local parsing result
    for(int i=1;i<parsing_results.size();i++){
        cv::Mat temp_parsing=parsing_results[i];

        int temp_height=temp_parsing.rows;
        int temp_width=temp_parsing.cols;
        int temp_x1=msrcnn_bbox[i-1][0];
        int temp_y1=msrcnn_bbox[i-1][1];

        cv::Mat temp_roi=global_temp(Rect(temp_x1,temp_y1,temp_width,temp_height));
        // now only simply add the results to fuse global and local
        temp_roi=temp_roi+temp_parsing;
    }
    global_parsing=global_parsing+global_temp;
    global_parsing=(global_parsing>0)/255;
    
    // merge the parsing and instance segmentation information
    global_parsing=global_parsing.mul(instance_masks);

    return global_parsing;
}


/**************************************************************************
    Author: Qiu Guangyue
    Date: 2020-08-18
    Function Name: res_pos()
    Function Description： 
        Definition of res_pos().
        Get the position x of every instance's torso from left to right.
    inputs: cv::Mat, int
    outputs: 
        std::map<int,int>
**************************************************************************/
std::map<int,int> res_pos(cv::Mat result_frame, int person_idx){
    std::map<int,int> pos;
    // Traverse the parsing result, and store the non-zero value's position
    for(int i=0;i<result_frame.cols;i++){
        for(int j=0;j<result_frame.rows;j++){
            auto cur_value=result_frame.at<uchar>(j, i);
            if(cur_value!=0){
                // If the non-zero vlaue hasn't appeared before, store it.
                if(pos.find(cur_value)==pos.end()){
                    pos[cur_value]=i;
                }
            }
        }
    }
    return pos;
}