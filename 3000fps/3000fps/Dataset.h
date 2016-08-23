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
	std::vector<double> radius;	//��ȡ�ľֲ������İ뾶

	int feat_num;
	int tree_num;			//ÿ��ɭ�ֵ���������
	int tree_max_depth;		//����������

};

/*****************         Class Sample          **********************/
class Image
{
public:
	cv::Mat_<uchar> image_gray;
	cv::Mat_<double> shape;
	cv::Mat_<double> current_shape;

	cv::Mat_<double> dS;				//��ŵ�ǰ����״����
	cv::Mat_<double> affine_mat;		//��meanshape�任��current_shape�ķ���任����  A * meanshape + b = current_shape��b������
	cv::Rect2f bbox;

	double err;							//ƽ�����
	std::vector<cv::Mat_<double> > shape_inter;		//��stage��״

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