#include "opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <memory>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int ac, char** av)
{
	int capture_width = 1000;
	int capture_height = 550;
	int margin = 50;
	int roi_long = 400;
	int left = margin;
	int right = left + roi_long;
	int top = margin;
	int bottom = top + roi_long;
	int roi_left = 190;
	int roi_top = 190;

	string pred;

	Mat frame;
	Mat resized_image;
	Mat roi;
	Mat gray;
	Mat img_threshold;
	Mat digit;
	Mat reshape;
	Mat unsqueeze;

	VideoCapture cap(0);

	// error : camera open
	if (!cap.isOpened())
	{
		cerr << "Camera open failed!" << endl;
		return -1;
	}

	torch::jit::script::Module module;
	try 
	{
		module = torch::jit::load("mnist_model_script.pt");
	}
	catch (const c10::Error& e)
	{
		std::cerr << "error loading the model \n";
		return -1;
	}

	// camera
	while (1)
	{
		cap >> frame;

		// resize frame ( frame -> resized_image )
		resize(frame, resized_image, Size(capture_width, capture_height));

		// add rectable to frame
		rectangle(resized_image, Rect(roi_left, roi_top, 170, 170), Scalar(0, 255, 0), 2);

		// cut ROI ( roi : 관심영역 )
		roi = resized_image(Rect(roi_left, roi_top, 170, 170));

		// Color change : RGB to GRAY ( roi -> gray )
		cvtColor(roi, gray, COLOR_RGB2GRAY);

		// threshold 임계값 처리 ( gray -> img_threshold )
		// THRESH_BINARY_INV or THRESH_BINARY
		threshold(gray, img_threshold, 150, 255, THRESH_BINARY_INV);

		// predict
		resize(img_threshold, digit, Size(28, 28));

		torch::Tensor input_tensor = torch::from_blob(digit.data, { 28, 28, 1 }, torch::kByte);

		input_tensor = input_tensor.permute({ 2, 0, 1 });
		input_tensor = input_tensor.toType(at::kFloat);

		vector<torch::jit::IValue> inputs;
		inputs.push_back(torch::unsqueeze(input_tensor, 0));
		at::Tensor output = module.forward(inputs).toTensor();

		int pred = torch::argmax(output, 1).item<int>();

		// text
		//putText(resized_image, to_string(pred), Point(left, top - 7), 0, 1, Scalar(0, 255, 0), 1, 8);
		putText(resized_image, "Result", Point(right + margin*3, margin*3), 0, 3, Scalar(0, 255, 0), 3, 8);
		putText(resized_image, to_string(pred), Point(right + margin * 5 - 20, margin * 7 + 20), 0, 6, Scalar(0, 255, 0), 5, 8);

		// camera
		imshow("camera", resized_image);
		imshow("threshold", img_threshold); 

		// if the user pressed "q", then stop
		if (waitKey(1) == 27)
		{
			printf("Exit \n");
			break;
		}
	}
	return 0;
}
