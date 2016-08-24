#ifndef _HEADERS_H_
#define _HEADERS_H_

#include "Dataset.h"
#include "model.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>

cv::Mat_<double> Center_and_scale(const Image &image);
cv::Mat_<double> Center_and_scale(const cv::Mat_<double> meanshape, const cv::Rect targetbbox);
cv::Mat_<double> Center_and_scale(const cv::Rect bbox, const cv::Mat_<double> shape);
cv::Mat_<double> Get_Affine_Mat(cv::Mat_<double> &X, cv::Mat_<double> &Y);
cv::Point2f get_ith_vector(cv::Mat_<double> &shape, int i);
void Draw_shapes(cv::Mat_<uchar> &image, const cv::Mat_<double> &shape);
void Draw_shapes(cv::Mat_<uchar> &image, const cv::Mat_<double> &shape, const cv::Rect bbox);
void Draw_shapes(Image &image);
void Round_shape(const cv::Mat_<uchar> &image, cv::Mat_<double> &S);
void facedetect(const cv::Mat_<uchar> &image, std::vector<cv::Rect> &bboxs);
Image Flip_Image(Image &image);
Image Rotate_Image(Image &image);
cv::Rect get_outerbox(cv::Mat_<double> &shape);
#endif