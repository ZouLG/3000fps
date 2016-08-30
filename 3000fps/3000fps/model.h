#ifndef _MODEL_H_
#define _MODEL_H_

#include "Dataset.h"
#include "liblinear/linear.h"
#include <opencv2/opencv.hpp>
#include <vector>

typedef struct Pixel_Pair
{
	cv::Point2f pixel1;
	cv::Point2f pixel2;
} Pair;						//�������ز��һ�����ص������LandMark��λ��

/***********************************    Class RandForest    ********************************/
class treenode
{
public:
	int threshold;	//���ѽڵ����ֵ
	int attr_index;	//�ýڵ㴦�������ԣ�����һ�����ز�������indexȥModel�е�pixel_pair���ҵ���Ӧ�����ز���

	bool isleaf;	//�Ƿ���Ҷ�ӽڵ�
	int leafnum;	//Ҷ�ӽڵ���ţ���Ҷ�ӽڵ�Ϊ-1

	int depth;
	treenode *lchild;
	treenode *rchild;
	treenode();
	treenode(int t, int a, int i, int l, int d, treenode *lc, treenode *rc);
};


class randforest
{
public:
	Parameter param;
	int landmark_index;
	std::vector<treenode *> roots;


	std::vector<int> Rand(int X, int x, bool flag);
	void Trainforest(Dataset &dataset, std::vector<Pixel_Pair> &pixel_pairs);
	cv::Mat_<int> generate_pixel_diff(Dataset &dataset, std::vector<Pixel_Pair> &pixel_pair);
	treenode* Train_SingleTree(Dataset &dataset, cv::Mat_<int> &pixel_diff, std::vector<int> &index_data, std::vector<int> &index_attr, int currentdepth, int &leafnum);
	void find_split_attr(Dataset &dataset, cv::Mat_<int> &pixel_diff, std::vector<int> &index_data, std::vector<int> &index_attr, int &split_attr,
						 int &thres, std::vector<int> &left_data_index, std::vector<int> &right_data_index);

	void BFS_Save_Trees(const std::string &rf_file);		//�������ɭ��
	void BFS_Load_Trees(const std::string &rf_file);		//�������ɭ��

};

/*****************         Class Regressor           *****************/
class Regressor
{
public:
	Parameter param;
	int stage;
	std::vector<std::vector<Pair> > pixel_pair;		//68*500
	std::vector<randforest> rf;						//68
	std::vector<model *> Model_x;
	std::vector<model *> Model_y;

	//cv::Mat_<double> Train(Dataset &dataset);
	void generate_pixel_pairs();
	void Get_LBF(Image &image, feature_node *x);
	void Save_Regressor(const std::string DirectoryPath);
	void Load_Regressor(const std::string RegressorDirectory);
};


/*****************         Class Model           *****************/
class Model
{
public:
	Parameter param;
	int current_stage;
	cv::Mat_<double> meanshape;
	std::vector<Regressor> regress;


	void Train(Dataset &dataset);
	void Test(Image &image);
	void TestCamera();

	void Save_Model(const std::string Model_Path = ".");
	void Load_Model(const std::string Model_path = "./Model");
};
#endif