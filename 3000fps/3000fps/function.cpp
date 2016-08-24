#include <opencv2/opencv.hpp>
#include "Dataset.h"
#include "model.h";
#include <ctime>

cv::Point2f get_ith_vector(cv::Mat_<double> &shape, int i)
{
	return (cv::Point2f(shape(i, 0), shape(i, 1)));
}

cv::Mat_<double> Get_Affine_Mat(cv::Mat_<double> &X, cv::Mat_<double> &Y)
// Y_i = A * X_i + b,求仿射变换A和b
// [ u ]   [ a0 b0 c0 ]            T   
// [ v ] = [ a1 b1 c1 ] * [ x y 1 ]
{
	cv::Mat_<double> inv, Xtemp_t, Y_t, Xtemp(X.cols + 1, X.rows);
	cv::Mat_<double> A;
	for (int i = 0; i < X.rows; i++)
	{
		Xtemp(0, i) = X(i, 0);
		Xtemp(1, i) = X(i, 1);
		Xtemp(2, i) = 1.0;
	}
	cv::transpose(Xtemp, Xtemp_t);
	cv::transpose(Y, Y_t);
	cv::invert(Xtemp*Xtemp_t, inv);
	A = Y_t * Xtemp_t * inv;
	return A.colRange(0, 2);
}

void Round_shape(const cv::Mat_<uchar> &image_gray, cv::Mat_<double> &S)
{
	for (int i = 0; i < S.rows; i++)
	{
		S(i, 0) = std::max(0, std::min((int)S(i, 0), image_gray.cols - 1));
		S(i, 1) = std::max(0, std::min((int)S(i, 1), image_gray.rows - 1));
	}
}

void Draw_shapes(cv::Mat_<uchar> &image, const cv::Mat_<double> &shape)
{
	int pointvalue = 255;
	cv::Mat_<uchar> tmp = image.clone();
	std::string Windowname = "window 1";
	cv::namedWindow(Windowname, cv::WINDOW_AUTOSIZE);
	for (int i = 0; i < shape.rows; i++)
	{
		//image((int)shape(i, 1), (int)shape(i, 0)) = pointvalue;    //点的坐标在图片中引用是y在前
		circle(tmp, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
	}
	imshow(Windowname, tmp);
}

void Draw_shapes(cv::Mat_<uchar> &image, const cv::Mat_<double> &shape, const cv::Rect bbox)
{
	int pointvalue = 255;
	cv::Mat_<uchar> tmp = image.clone();
	std::string Windowname = "window 1";
	cv::rectangle(tmp, bbox, 127);
	cv::namedWindow(Windowname, cv::WINDOW_AUTOSIZE);
	for (int i = 0; i < shape.rows; i++)
	{
		//image((int)shape(i, 1), (int)shape(i, 0)) = pointvalue;    //点的坐标在图片中引用是y在前
		circle(tmp, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
	}
	imshow(Windowname, tmp);
}

void Draw_shapes(Image &image)
{
	int pointvalue = 255;
	std::string Windowname = "window 1";
	cv::Mat_<uchar> temp = image.image_gray.clone();
	rectangle(temp, image.bbox, 255);

	cv::namedWindow(Windowname, cv::WINDOW_AUTOSIZE);
	for (int i = 0; i < image.shape.rows; i++)
	{
		//temp(image.shape(i, 1), image.shape(i, 0)) = pointvalue;    //点的坐标在图片中引用是y在前
		circle(temp, cv::Point2f(image.shape(i, 0), image.shape(i, 1)), 2, (255));
		//char txt[3];
		//_itoa(i, txt, 10);
		//std::string text = txt;
		//cv::putText(temp, text, cv::Point(image.shape(i, 0), image.shape(i, 1)), cv::FONT_HERSHEY_TRIPLEX, 0.4, 120);
	}
	imshow(Windowname, temp);
}


cv::Mat_<double> Center_and_scale(const Image &image)	//归一化
{
	cv::Mat_<double> temp(image.shape.size());
	for (int i = 0; i < image.shape.rows; i++)
	{
		temp(i, 0) = (image.shape(i, 0) - image.bbox.x - image.bbox.width / 2) / image.bbox.width;
		temp(i, 1) = (image.shape(i, 1) - image.bbox.y - image.bbox.height / 2) / image.bbox.height;
	}
	return temp;
}

cv::Mat_<double> Center_and_scale(const cv::Rect bbox, const cv::Mat_<double> shape)	//缩小
{
	cv::Mat_<double> shape_(shape.size());
	double center_x = bbox.x + bbox.width / 2;
	double center_y = bbox.y + bbox.height / 2;
	for (int i = 0; i < shape.rows; i++)
	{
		shape_(i, 0) = (shape(i, 0) - center_x) / bbox.width;
		shape_(i, 1) = (shape(i, 1) - center_y) / bbox.height;
	}
	return shape_;
}

cv::Mat_<double> Center_and_scale(const cv::Mat_<double> meanshape, const cv::Rect targetbbox)	//放大
{
	cv::Mat_<double> shape(meanshape.size());
	double center_x = targetbbox.x + targetbbox.width/2;
	double center_y = targetbbox.y + targetbbox.height/2;
	for (int i = 0; i < meanshape.rows; i++)
	{
		shape(i, 0) = meanshape(i, 0) * targetbbox.width + center_x;
		shape(i, 1) = meanshape(i, 1) * targetbbox.height + center_y;
	}
	return shape;
}

Image Flip_Image(Image &image)
{
	Image image_flip;
	int FlipArray[] = { 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0, 26, 25, 24, 23, 22, 21,
						20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36,
						41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65 };
	image_flip.shape.create(image.shape.size());
	flip(image.image_gray, image_flip.image_gray, 1);
	for (int i = 0; i < image.shape.rows; i++)
	{
		image_flip.shape(i, 0) = image.image_gray.cols - image.shape(FlipArray[i], 0);
		image_flip.shape(i, 1) = image.shape(FlipArray[i], 1);
	}
	return image_flip;
}

Image Rotate_Image(Image &image)
{
	Image image_rotate;
	cv::RNG rg(time(0));
	cv::Mat_<double> tmp = image.shape.row(28) - image.shape.row(9);
	double th1, th = atan(tmp(0, 1) / tmp(0.0)) * 180 / CV_PI - 90;
	do{
		th1 = rg.uniform(-12.0, 12.0);
	} while (th1 - th < 5 && th1 - th > -5);

	cv::Mat_<double> rotate_mat = cv::getRotationMatrix2D(cv::Point2f(image.image_gray.cols / 2, image.image_gray.rows / 2), th1, 1);
	warpAffine(image.image_gray, image_rotate.image_gray, rotate_mat, image.image_gray.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

	tmp(0, 0) = image.image_gray.cols / 2; tmp(0, 1) = image.image_gray.rows / 2;
	resize(tmp, tmp, cv::Size(2, 68), 0, 0, CV_INTER_NN);
	cv::Mat_<double> temp;
	cv::transpose(rotate_mat.colRange(0, 2), temp);
	image_rotate.shape = (image.shape - tmp) * temp + tmp;
	return image_rotate;
}

cv::Rect get_outerbox(cv::Mat_<double> &shape)
{
	double xmin = shape(0, 0), ymin = shape(0, 1);
	double xmax = xmin, ymax = ymin;
	for (int i = 0; i < shape.rows; i++)
	{
		if (shape(i, 0) < xmin)
			xmin = shape(i, 0);
		else
		{
			if (shape(i, 0)>xmax)
				xmax = shape(i, 0);
		}

		if (shape(i, 1) < ymin)
			ymin = shape(i, 1);
		else
		{
			if (shape(i, 1)>ymax)
				ymax = shape(i, 1);
		}
	}
	return cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
}