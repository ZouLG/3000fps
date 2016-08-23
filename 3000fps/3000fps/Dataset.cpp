#include "headers.h"
#include <fstream>

// 在dataset路径下通过控制台输入dir /b/s/p/w *.jpg>Path_Images.txt保存所有jpg图片的路径
int Dataset::Load_Dataset(std::string datasetpath, bool flipflag)
{
	std::cout << "Loading Dataset..." << std::endl;
	std::ifstream fp1, fp3;
	datasetsize = 0;
	fp1.open(datasetpath + "/Path_Images.txt");
	//fp2.open(datasetpath + "/facebbox.txt");
	if (!fp1.good())
		return 0;
	while (!fp1.eof())
	{
		std::string im_path, temp;
		//Rect2f Rect_tmp;
		//Mat_<uchar> im_tmp;
		Image im_tmp;
		getline(fp1, im_path);
		if (im_path.empty())
			continue;

		im_tmp.image_gray = cv::imread(im_path, cv::IMREAD_GRAYSCALE);

		*(im_path.end() - 1) = 's'; *(im_path.end() - 2) = 't'; *(im_path.end() - 3) = 'p';
		fp3.open(im_path);
		if (!fp3.good())
			continue;
		getline(fp3, temp); getline(fp3, temp); getline(fp3, temp);
		im_tmp.shape.create(param.landmark_num, 2);
		for (int i = 0; i < param.landmark_num; i++)
		{
			fp3 >> im_tmp.shape(i, 0) >> im_tmp.shape(i, 1);
		}
		fp3.close();

		if (im_tmp.image_gray.cols > 1800 || im_tmp.image_gray.rows > 1800){
			cv::resize(im_tmp.image_gray, im_tmp.image_gray, cv::Size(im_tmp.image_gray.cols / 4, im_tmp.image_gray.rows / 4), 0, 0, cv::INTER_LINEAR);
			im_tmp.shape = im_tmp.shape / 4.0;
		}
		else if (im_tmp.image_gray.cols > 1000 || im_tmp.image_gray.rows > 1000){
			cv::resize(im_tmp.image_gray, im_tmp.image_gray, cv::Size(im_tmp.image_gray.cols / 2, im_tmp.image_gray.rows / 2), 0, 0, cv::INTER_LINEAR);
			im_tmp.shape = im_tmp.shape / 2.0;
		}

		if (Get_bbox(im_tmp))		//找到包含landmark的bbox
		{
			Enlarge_bbox(im_tmp);
			Round_shape(im_tmp.image_gray, im_tmp.shape);
			images.push_back(im_tmp);
			datasetsize++;
		}

		if (flipflag)
		{
			Image im_tmp_flip = Flip_Image(im_tmp);
			if (Get_bbox(im_tmp_flip))
			{
				images.push_back(im_tmp_flip);
				datasetsize++;
			}
		}
		/******************************************/
	}
	fp1.close();
	std::cout << datasetsize << " images totally" << std::endl;
	return 1;
}


bool Dataset::Get_bbox(Image &image)
{
	std::vector<cv::Rect> bboxs;
	facedetect(image.image_gray, bboxs);
	cv::Rect bbox, outerbox = get_outerbox(image.shape);

	double ratio = 0.0;
	for (int i = 0; i < bboxs.size(); i++)
	{
		double tmp = (double)(bboxs[i] & outerbox).area() / (double)(bboxs[i] | outerbox).area();
		if (tmp > ratio)
		{
			ratio = tmp;
			bbox = bboxs[i];
		}
	}
	if (ratio > 0.5)
	{
		image.bbox = bbox;
		return true;
	}
	else
		return false;
}

void Dataset::Enlarge_bbox(Image &image)
{
	float dx = std::max(-image.bbox.width / 2, -image.bbox.x), dy = std::max(-image.bbox.height / 2, -image.bbox.y);
	cv::Point2f p(dx, dy);
	cv::Rect2f temp = image.bbox + p;

	if (!image.shape.empty())
	{
		for (int i = 0; i < image.shape.rows; i++)
		{
			image.shape(i, 0) -= temp.x;
			image.shape(i, 1) -= temp.y;
		}
	}
	float dw = std::min(-2 * dx, image.image_gray.cols - temp.x - image.bbox.width - 1);
	float dh = std::min(-2 * dy, image.image_gray.rows - temp.y - image.bbox.height - 1);
	temp = temp + cv::Size2f(dw, dh);
	image.image_gray = image.image_gray(temp);
	image.bbox = image.bbox - cv::Point2f(temp.x, temp.y);
}


void Dataset::Get_Meanshape()
{
	meanshape = cv::Mat::zeros(param.landmark_num, 2, CV_32FC1);
	for (int i = 0; i < images.size(); i++)
	{
		meanshape += Center_and_scale(images[i]);
		//meanshape += Center_and_scale(images[i].bbox, images[i].current_shape);
	}
	meanshape = meanshape / images.size();
}

void Dataset::Set_S0(int mode)
{
	cv::Mat_<double> init;
	cv::RNG rg(time(0));
	for (int i = 0; i < datasetsize; i++)
	{
		if (mode == 1)
		{
			int n;
			do{
				n = std::round(rg.uniform(0, datasetsize - 1));
			} while (n == i);
			init = Center_and_scale(images[n]);
		}
		else
			init = meanshape;

		images[i].current_shape = Center_and_scale(init, images[i].bbox);
		images[i].affine_mat = Get_Affine_Mat(meanshape, images[i].current_shape);
		Round_shape(images[i].image_gray, images[i].current_shape);
		images[i].dS = Center_and_scale(images[i].bbox, images[i].shape) - Center_and_scale(images[i].bbox, images[i].current_shape);
	}
}

void Dataset::Show_set(int time_ms)
{
	for (int i = 0; i < datasetsize; i++)
	{
		Draw_shapes(images[i]);
		cv::waitKey(time_ms);
	}
}

double Dataset::Cal_Error()
{
	double sum_err = 0.0;
	for (int i = 0; i < datasetsize; i++)
	{
		images[i].err = 0.0;
		for (int j = 0; j < images[i].dS.rows; j++)
		{
			images[i].err += std::pow(images[i].dS(j, 0), 2);
			images[i].err += std::pow(images[i].dS(j, 1), 2);
		}
		sum_err += images[i].err;
	}
	std::cout << "Total normalized error: " << sum_err <<
		"	Mean error: " << sum_err / datasetsize << std::endl;
	return sum_err;
}
