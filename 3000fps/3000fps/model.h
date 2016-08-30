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
} Pair;						//计算像素差的一对像素点相对于LandMark的位置

/***********************************    Class RandForest    ********************************/
class treenode
{
public:
	int threshold;	//分裂节点的阈值
	int attr_index;	//该节点处分裂属性（是哪一对像素差），用这个index去Model中的pixel_pair中找到对应的像素差点对

	bool isleaf;	//是否是叶子节点
	int leafnum;	//叶子节点序号，非叶子节点为-1

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

	void BFS_Save_Trees(const std::string &rf_file);		//保存随机森林
	void BFS_Load_Trees(const std::string &rf_file);		//加载随机森林

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