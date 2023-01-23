// One-stop header.
#include <torch/script.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <boost/asio.hpp>

using namespace boost::asio;

#define kIMAGE_SIZE 224
#define kCHANNELS 3
#define kTOP_K 3

// Function to handle the image path
void handleImage(std::string path) {
    std::cout << "Received image path: " << path << std::endl;
    // Add your code here to handle the image
}


bool LoadImage(std::string file_name, cv::Mat &image) {
  image = cv::imread(file_name);  // CV_8UC3
  if (image.empty() || !image.data) {
    return false;
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  std::cout << "== image size: " << image.size() << " ==" << std::endl;

  // scale image to fit
  cv::Size scale(kIMAGE_SIZE, kIMAGE_SIZE);
  cv::resize(image, image, scale);
  std::cout << "== simply resize: " << image.size() << " ==" << std::endl;

  // convert [unsigned int] to [float]
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

  return true;
}

bool LoadImageNetLabel(std::string file_name,
                       std::vector<std::string> &labels) {
  std::ifstream ifs(file_name);
  if (!ifs) {
    return false;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    labels.push_back(line);
  }
  return true;
}

std::string model_name="../resnet18.pt";

std::string labels_txt="../label.txt";

int main(int argc, const char *argv[]) {
  // if (argc != 3) {
  //   std::cerr << "Usage: classifier <path-to-exported-script-module> "
  //                "<path-to-lable-file>"
  //             << std::endl;
  //   return -1;
  // }
  // Initialize the server
    io_service service;
    ip::tcp::endpoint ep(ip::tcp::v4(), 12345);
    ip::tcp::acceptor acc(service, ep);
    cv::Mat image;
    torch::jit::script::Module module = torch::jit::load(model_name);
  
    std::cout << "== Model [" << model_name << "] loaded!\n";
    std::vector<std::string> labels;
    if (LoadImageNetLabel(labels_txt, labels)) {
      std::cout << "== Label loaded! Let's try it\n";
    } else {
      std::cerr << "Please check your label file path." << std::endl;
      return -1;
    }

    while (true) {
        // Wait for a client to connect
        ip::tcp::socket sock(service);
        acc.accept(sock);

        // Read the incoming message
        boost::asio::streambuf buf;
        boost::asio::read_until(sock, buf, "\r\n\r\n");
        std::string message = boost::asio::buffer_cast<const char*>(buf.data());

        // Check for a "POST" command
        if (message.find("POST") == 0) {
            // Extract the image path from the message body
            std::string imagePath = message.substr(message.find("\r\n\r\n") + 4);
            imagePath = imagePath.substr(0, imagePath.find("\r\n"));
            
            // Send the image path to the function
            handleImage(imagePath);
            if (LoadImage(imagePath, image)) {
            auto input_tensor = torch::from_blob(
          image.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
          input_tensor = input_tensor.permute({0, 3, 1, 2});
          input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
          input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
          input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);



          torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();

          auto results = out_tensor.sort(-1, true);
          auto softmaxs = std::get<0>(results)[0].softmax(0);
          auto indexs = std::get<1>(results)[0];
      
        auto idx = indexs[0].item<int>();
        std::cout << "    Label:  " << labels[idx] << std::endl;
        std::cout << "    With Probability:  "
                  << softmaxs[0].item<float>() * 100.0f << "%" << std::endl;
          std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nClass: " +  labels[idx] + "\r\n";
            boost::asio::write(sock, boost::asio::buffer(response));
        } else {
            // Send an error message if the command is not "POST"
            std::string response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\r\nInvalid command\r\n";
            boost::asio::write(sock, boost::asio::buffer(response));
        }

    } else {
      std::cout << "Can't load the image, please check your path." << std::endl;
    }
            // Send the image path back to the client
          
    }

    return 0;
  }
  




