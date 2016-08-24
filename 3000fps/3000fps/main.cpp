#include <opencv2/opencv.hpp>
#include "liblinear/linear.h"
#include "headers.h"
#include <string>
#pragma comment(lib,"libfacedetect-x64.lib")

void main()
{
	bool flag = true;	//true 训练 ,false 测试
	if (flag){
		Parameter param;
		param.landmark_num = 68;
		param.stage_num = 6;
		param.radius = { 0.16, 0.11, 0.08, 0.06, 0.04, 0.02, 0.01, 0.005 };
		param.feat_num = 500;
		param.tree_num = 12;
		param.tree_max_depth = 5;

		Dataset data;
		data.param = param;
		std::string datasetpath = "D:/Projects_Face_Detection/Datasets/helen/trainset";
		data.Load_Dataset(datasetpath, true, true);	//用翻转和随机旋转的方式扩展训练集
		//data.Show_set();
		data.Get_Meanshape();
		data.Set_S0();
		data.Cal_Error();

		Model m;
		m.param = param;
		m.Train(data);
		m.Save_Model();

		for (int i = 0; i < 10; i++)
		{
			Image test = Image();
			test.image_gray = data.images[i].image_gray;
			m.Test(test);
		}

	}
	else{
		Model m;
		m.Load_Model();
		std::ifstream fp;
		fp.open("D:/Projects_Face_Detection/Datasets/helen/testset/Path_Images.txt");
		for (int i = 0; i < 100; i++)
		{
			std::string impath;
			getline(fp, impath);
			if (i % 4 != 0)
				continue;

			Image im = Image();
			im.image_gray = cv::imread(impath, cv::IMREAD_GRAYSCALE);
			if (im.image_gray.cols > 1800){
				cv::resize(im.image_gray, im.image_gray, cv::Size(im.image_gray.cols / 4, im.image_gray.rows / 4), 0, 0, cv::INTER_LINEAR);
			}
			else if (im.image_gray.cols > 1100){
				cv::resize(im.image_gray, im.image_gray, cv::Size(im.image_gray.cols / 2, im.image_gray.rows / 2), 0, 0, cv::INTER_LINEAR);
			}
			m.Test(im);
		}
		fp.close();


		Image im = Image();
		im.image_gray = cv::imread("test.jpg", cv::IMREAD_GRAYSCALE);
		m.Test(im);
		Draw_shapes(im.image_gray, im.current_shape);
		cv::waitKey(0);

		/****************************************************************/
		/*cv::VideoCapture cap;
		cap.open(0);
		if (!cap.isOpened())
		{
		std::cerr << "Camera open failed..." << std::endl;
		exit(0);
		}

		while (1)
		{
		Image mtest = Image();
		cv::Mat temp;
		cap.read(temp);
		//cv::namedWindow("a", cv::WINDOW_AUTOSIZE);
		//cv::imshow("a", temp);
		//cv::waitKey(0);
		cv::cvtColor(temp, mtest.image_gray, CV_RGB2GRAY);
		//mtest.image_gray = imread("zhang1.jpg", cv::IMREAD_GRAYSCALE);
		m1.Test(mtest);
		Draw_shapes(mtest.image_gray, mtest.current_shape);
		cv::waitKey(1);
		}*/

	}




}
