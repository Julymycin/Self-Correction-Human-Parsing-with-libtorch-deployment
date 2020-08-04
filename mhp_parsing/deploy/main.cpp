#include "de2.h"
#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>

using namespace std;
using namespace caffe2;
using namespace cv;

void Mask_RCNN::load_net(std::string init_model_file, std::string predict_model_file)
{
    CAFFE_ENFORCE(ReadProtoFromFile(init_model_file, &initNet_));
    CAFFE_ENFORCE(ReadProtoFromFile(predict_model_file, &predictNet_));
    CAFFE_ENFORCE(workSpace.RunNetOnce(initNet_));
}

ins_pre Mask_RCNN::run(cv::Mat &input)
{
    de2::ins_pre pre;

    int height = input.rows;
    int width = input.cols;

    int img_size[2] = {input.rows, input.cols};
    pre.image_size = img_size;

    Mat dst1;
    cvtColor(input, dst1, CV_BGR2GRAY);
    cvtColor(dst1, input, CV_GRAY2BGR);

    for (auto &str : predictNet_.external_input())
    {
        workSpace.CreateBlob(str);
    }
    //Workspace workSpace;
    //	CAFFE_ENFORCE(workSpace.CreateNet(predictNet_));

    // setup inputs
    auto data = BlobGetMutableTensor(workSpace.GetBlob("data"), caffe2::CPU);
    data->Resize(1, 3, height, width);
    float *ptr = data->mutable_data<float>();
    // HWC to CHW
    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < height * width; ++i)
        {
            ptr[c * height * width + i] = static_cast<float>(input.data[3 * i + c]);
        }
    }

    auto im_info =
        BlobGetMutableTensor(workSpace.GetBlob("im_info"), caffe2::CPU);
    im_info->Resize(1, 3);
    float *im_info_ptr = im_info->mutable_data<float>();
    im_info_ptr[0] = height;
    im_info_ptr[1] = width;
    im_info_ptr[2] = 1.0;
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

    pre.pred_boxes = bbox;
    pre.scores = scores;
    pred.pred_classes = labels;
    pred.masks = mask_probs;

    return pred;
}

// make crop and mask w mask nms
img_item mcmm(ins_pre pred, cv::Mat &input)
{
    int img_h = pred.image_size.data<int>()[0];
    int img_w = pred.image_size.data<int>()[1];
    float exp_ratio = 1.2;
    Mat panoptic_seg = Mat::ones(img_h, img_w);
    int num_instances = pred.bbox.sizes()[0];
    std::list<float> exp_bbox;
    std::list<int> ori_bbox;
    std::list<float> bbox_score_list;
    if (num_instances == 0)
    {
        cout << "no instances" << endl;
    }
    int person_idx = 0;
    for (int i = 0; i < num_instances; ++i)
    {
        float score = pred.scores.data<float>()[i];
        int cls = pred.pred_classes.data<int>()[i];
        if (score < 0.6 || cls != 0)
            continue; // skip them
        Mat mask = pred.pred_masks.data<float>[i];
        float mask_area = cv::sum(mask)[0];
        if (mask_area == 0)
            continue;
        Mat intersetc = (mask > 0) & (panoptic_seg > 0);
        float intersect_area = cv::sum(intersetc)[0];
        if (intersect_area * 1.0 / mask_area > 0.5)
            continue;
        if (intersect_area > 0)
            mask = mask & (panoptic_seg == 0);
        person_idx += 1;
        Mat dist1 = mask > 0;
        dist1 = dist1 / 255 * person_idx;
        panoptic_seg = panoptic_seg | dist1;

        bbox_score_list.push_back(score);
        float ins_bbox[4] = pred.pred_boxes.data<float>()[i];
        float x_min = ins_bbox[0];
        float y_min = ins_bbox[1];
        float x_max = ins_bbox[2];
        float y_max = ins_bbox[3];
        float exp_x = (x_max - x_min) * ((exp_ratio - 1) / 2);
        float exp_y = (y_max - y_min) * ((exp_ratio - 1) / 2);
        int new_x_min = max(0, round(x_min - exp_x));
        int new_y_min = max(0, round(y_min - exp_y));
        int new_x_max = min(img_w - 1, round(x_max + exp_x));
        int new_y_max = min(img_h - 1, round(y_max + exp_y));
        exp_bbox.push_back({new_x_min, new_y_min, new_x_max, new_y_max});
        ori_bbox.push_back({x_min, y_min, x_max, y_max});
    }
    de2::img_item item;
    item.img_height = img_h;
    item.img_weight = img_w;
    item.center = {img_h / 2.0, img_w / 2.0};
    item.person_num = person_idx;
    item.person_bbox = exp_bbox;
    item.real_person_bbox = ori_bbox;
    item.person_bbox_score = bbox_score_list;
    item.instance_masks = panoptic_seg;

    return item;
}

//global and local evaluate
img_item gl_parsing(img_item item, cv::Mat &input)
{
    std::vector<cv::Mat> parsing_results;
    std::string ckpt_path = "/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/pascal_abn.pt";
    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // module = torch::jit::load(argv[1]);
        module = torch::jit::load(model);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return torch::zeros(1);
    }
    int input_size = 473;
    cv::Mat img;
    for (int i = 0; i <= item.preson_num; i++)
    {
        if (i == 0)
        {
            img = input;
        }
        else
        {
            int x_min = item.person_bbox[i - 1][0];
            int y_min = item.person_bbox[i - 1][1];
            int temp_w = item.person_bbox[i - 1][2] - x_min;
            int temp_h = item.person_bbox[i - 1][3] - y_min;
            Rect rect(x_min, y_min, temp_w, temp_h);
            img = input(rect);
        }
        int height = img.rows;
        int width = img.cols;
        float c[2] = {height / 2.0, width / 2.0};
        float s[2] = {max(height, width) - 1, max(height, width) - 1};
        int r = 0;
        //
        //

        cv::Mat img_float;
        img.convertTo(img_float, CV_32F, 1.0 / 255);
        cv::resize(img_float, img_float, cv::Size(input_size, input_size));
        auto img_tensor = torch::from_blob(img_float.data, {1, 3, input_size, input_size});

        img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
        img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
        img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);

        std::vector<torch::jit::IValue> inputs;

        inputs.push_back(img_tensor.to(at::kCUDA));

        at::Tensor output = module.forward(inputs).toTensor(); // Tensor [NUM_CLASSES, 128, 128]
        // output=torch::unsqueeze(output,0);
        auto res = torch::upsample_bilinear2d(output, {input_size, input_size}, true); // Tensor [NUM_CLASSES, input_s, input_s]
        res = torch::squeeze(res);                                                     // Tensor [NUM_CLASSES, input_s, input_s]
        res = res.permute({1, 2, 0});                                                  //CHW->HWC
        //
        //argmax
        //where==2
        //parsing_transform
        //
        res = res.to(torch::kCPU).squeeze().detach().to(torch::kU8);
        cv::Mat parsing_result(height, width, CV_8UC1);
        std::memcpy((void *)parsing_result.data, res.data_ptr(), sizeof(torch::kU8) * res.numel());
        parsing_results.push_back(parsing_result);
    }
    item.parsing_results=parsing_results;
}

cv::Mat parsing_fusion(img_item item){
    int img_height=item.img_height;
    int img_width=item.img_width;
    std::vector<std::vector<int>> msrcnn_bbox=item.person_bbox;
    std::vector<float> bbox_score=item.person_bbox_score;
    std::vector<cv::Mat> parsing_results=item.parsing_results;
    std::vector<cv::Mat> instance_masks=item.instance_masks;

    cv::Mat global_parsing=parsing_results[0];
    cv::Mat global_temp(img_height, img_width, CV_8UC1);
    for(int i=1;i<parsing_results.size();i++){
        
        cv::Mat temp_parsing=parsing_results[i];
        int temp_height=temp_parsing.rows;
        int temp_width=temp_parsing.cols;
        int temp_x1=msrcnn_bbox[0];
        int temp_y1=msrcnn_bbox[1];
        cv::Mat temp_roi=global_temp(Rect(temp_x1,temp_y1,temp_width,temp_height));
        temp_roi=temp_roi+temp_parsing;
    }
    global_parsing=global_parsing+global_temp;
    global_parsing=(global_parsing>0)/255;
    global_parsing=global_parsing.mul(instance_masks);

    return global_parsing;
}
void main()
{
}