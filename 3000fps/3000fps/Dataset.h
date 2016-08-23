#ifndef _DATASET_H_
#define _DATASET_H_
#include <opencv2/opencv.hpp>
#include <vector>

/*****************         Class Param           **********************/
class Parameter
{
public:
	int landmark_num;
	int stage_num;
	std::vector<double> radius;	//提取的局部特征的半径

	int feat_num;
	int tree_num;			//每个森林的树的数量
	int tree_max_depth;		//树的最大深度

};

/*****************         Class Sample          **********************/
class Image
{
public:
	cv::Mat_<uchar> image_gray;
	cv::Mat_<double> shape;
	cv::Mat_<double> current_shape;

	cv::Mat_<double> dS;				//存放当前的形状增量
	cv::Mat_<double> affine_mat;		//从meanshape变换到current_shape的仿射变换矩阵  A * meanshape + b = current_shape，b忽略了
	cv::Rect2f bbox;

	double err;							//平均误差
	std::vector<cv::Mat_<double> > shape_inter;		//各stage形状

};


/******************         Class Data              *********************/
class Dataset
{
public:
	int datasetsize;
	Parameter param;

	std::vector<Image> images;
	cv::Mat_<double> meanshape;

	bool Get_bbox(Image &image);

	int Load_Dataset(std::string datasetpath, bool flipflag = 0);

	void Enlarge_bbox(Image &image);

	void Get_Meanshape();

	void Set_S0(int mode = 0);

	void Show_set(int time_ms = 0);

	double Cal_Error();
};
#endif