#include "opencv.hpp"
#include "opencv_modules.hpp"
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <memory>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int ac, char** av)
{
	int capture_width = 900;
	int roi_long = 400;
	int margin = 50;
	int top = margin;
	int right = capture_width - margin;
	int bottom = top + roi_long;
	int left = right - roi_long;

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
		module = torch::jit::load("./mnist_model_realreal.pt");
	}
	catch (const c10::Error& e)
	{
		std::cerr << "error loading the model\n";
		return -1;
	}

	// camera
	while (1)
	{
		cap >> frame;

		// resize frame ( frame -> resized_image )
		resize(frame, resized_image, Size(900, 900));

		// add rectable to frame
		rectangle(resized_image, Rect(left, top, right, bottom), Scalar(0, 255, 0), 2);

		// cut ROI ( roi : ���ɿ��� )
		roi = resized_image(Rect(left + 2, top + 2, 396, 396));

		// Color change : RGB to GRAY ( roi -> gray )
		cvtColor(roi, gray, CV_RGB2GRAY);

		// threshold �Ӱ谪 ó�� ( gray -> img_threshold )
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
		putText(resized_image, to_string(pred), Point(left, top - 7), 0, 1, Scalar(0, 255, 0), 1, 8);

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