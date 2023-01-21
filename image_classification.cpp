#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <stdio.h>
#include <torch/script.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
  // Load pre-trained model
  torch::jit::script::Module model = torch::jit::load("mobilenet_v2-b0353104.pt");

  // Prepare image for classification
  if (argc != 2) {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread(argv[1], 1);
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }
  Mat image_tensor = image.clone().reshape(1, image.rows * image.cols * 3);
  torch::Tensor tensor_image = torch::from_blob(image_tensor.data, {1, image_tensor.total() * image_tensor.channels()});
  tensor_image = tensor_image.permute({0, 3, 1, 2});
  tensor_image = tensor_image.to(torch::kFloat);

  // Normalize image
  tensor_image[0][0] = tensor_image[0][0].sub(0.485).div(0.229);
  tensor_image[0][1] = tensor_image[0][1].sub(0.456).div(0.224);
  tensor_image[0][2] = tensor_image[0][2].sub(0.406).div(0.225);

  // Forward pass
  auto output = model.forward({tensor_image}).toTensor();

  // Get class with highest probability
  auto prob = torch::softmax(output, 1);
  torch::Tensor prob_max;
  torch::Tensor class_index;
  std::tie(prob_max, class_index) = prob.max(1);

  // Print class name
  int class_idx = class_index[0].item<int>();
  cout << "Class: " << class_idx << endl;

  return 0;
}
