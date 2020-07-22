#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>


at::Tensor infer(int t, std::string imgname) {
    std::string root_path="/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/";
    enum choices {lip, atr, pascal };
    std::map<choices, std::string> model_path;
    model_path[lip]="lip_abn.pt";
    model_path[atr]="atr_abn.pt";
    model_path[pascal]="pascal_abn.pt";
    int input_size=512;
    int num_cls=18;
    auto ity=choices(t); // dataset type
    // dataset_settings = {
//     'lip': {
//         'input_size': [473, 473],
//         'num_classes': 20,
//         'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
//                   'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
//                   'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
//     },
//     'atr': {
//         'input_size': [512, 512],
//         'num_classes': 18,
//         'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
//                   'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
//     },
//     'pascal': {
//         'input_size': [512, 512],
//         'num_classes': 7,
//         'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
//     }
// }
    std::string model=root_path+model_path[ity];
    switch (ity)
    {
    case lip:
        input_size=473;
        num_cls=20;
        break;
    case atr:       
        input_size=512;
        num_cls=18;
        break;
    case pascal:    
        input_size=512;
        num_cls=7;
        break;
    default:
        break;
    }

    cv::Mat img=cv::imread(imgname);
    cv::Mat img_float;
    img.convertTo(img_float, CV_32F, 1.0 / 255);
    cv::resize(img_float, img_float, cv::Size(input_size, input_size));
    auto img_tensor = torch::from_blob(img_float.data, { 1, input_size, input_size, 3 });
    img_tensor = img_tensor.permute({ 0, 3, 1, 2 });

    img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
    img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
    img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);

    torch::jit::script::Module module;
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
        // module = torch::jit::load(argv[1]);
        module = torch::jit::load(model);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return torch::zeros(1);
    }
    // std::cout<<model;

    // std::cout << "ok\n";

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, input_size, input_size}).to(at::kCUDA));
    inputs.push_back(img_tensor.to(at::kCUDA));

    // std::cout<<inputs<<'\n';

    // Execute the model and turn its output into a tensor.
    // auto output = module.forward(inputs).toTensor(); // Tensor [1, NUM_CLASSES, 128, 128]
    // auto res=torch::squeeze(output); // Tensor [NUM_CLASSES, 128, 128]
    at::Tensor output = module.forward(inputs).toTensor(); // Tensor [NUM_CLASSES, 128, 128]
    // output=torch::unsqueeze(output,0);
    auto res=torch::upsample_bilinear2d(output, {input_size,input_size}, true );  // Tensor [NUM_CLASSES, input_s, input_s]
    res=torch::squeeze(res); // Tensor [NUM_CLASSES, input_s, input_s]
    res=res.permute({1, 2, 0}); //CHW->HWC
    inputs.pop_back();
    return res;
}

int main(){
    std::string img_path="/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/example-app/15948686054672.png";
    auto output=infer(1,img_path); // Tensor [ inpuit_s, input_s, NUM_CLASSES]
    std::cout<<output.sizes();
    // auto sll=res.slice(0,0,1);
    // std::cout<<sll;
    return 0;
}
