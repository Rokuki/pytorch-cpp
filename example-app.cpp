#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  //module->to(at::kCUDA);

  assert(module != nullptr);
  std::cout << "ok\n";

  std::vector<torch::jit::IValue> inputs;

  cv::Mat image;
  image = cv::imread(argv[2], 1);
  cv::cvtColor(image, image, CV_BGR2RGB);
  cv::Mat img_float;
  image.convertTo(img_float, CV_32F, 1.0/255);
  cv::resize(img_float, img_float, cv::Size(224, 224));

  cout << "resize ok\n";

  //std::cout << img_float.at<cv::Vec3f>(56,34)[1] << std::endl;
  auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, 224, 224, 3});
  img_tensor = img_tensor.permute({0,3,1,2});
  cout << "permute ok\n";
  img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
  img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
  img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);


  auto img_var = torch::autograd::make_variable(img_tensor, false);
  inputs.push_back(img_var);

  // Create a vector of inputs.

  //inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module->forward(inputs).toTensor();
  auto max_result = output.max(1, true);
  auto max_index = std::get<1>(max_result).item<float>();

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  cout << max_index << endl;
}
