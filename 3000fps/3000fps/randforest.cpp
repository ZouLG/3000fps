#include "headers.h"
#include <queue>
#include "ctime"

treenode::treenode()
{
	threshold = 0;
	attr_index = 0;
	leafnum = 0;
	isleaf = true;
	depth = 1;
	lchild = NULL;
	rchild = NULL;
}

cv::Mat_<int> randforest::generate_pixel_diff(Dataset &dataset, std::vector<Pair> &pixel_pair)
{
	cv::Mat_<int> pixel_diff(dataset.datasetsize, param.feat_num);
#pragma omp parallel for
	for (int i = 0; i < dataset.datasetsize; i++)
	{
		cv::Mat_<double> A = Get_Affine_Mat(dataset.meanshape, dataset.images[i].current_shape);
		dataset.images[i].affine_mat = A;
		//cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		for (int j = 0; j < param.feat_num; j++)
		{
			cv::Point2f temp1 = pixel_pair[j].pixel1;
			cv::Point2f temp2 = pixel_pair[j].pixel2;

			cv::Mat_<double> tmp = A * cv::Mat_<double>(temp1);
			temp1 = cv::Point2f(tmp);
			tmp = A * cv::Mat_<double>(temp2);
			temp2 = cv::Point2f(tmp);

			cv::Point2f current_pos = get_ith_vector(dataset.images[i].current_shape, landmark_index);
			temp1 += current_pos;
			temp2 += current_pos;

			temp1.x = std::max(0, std::min((int)temp1.x, dataset.images[i].image_gray.cols - 1));
			temp1.y = std::max(0, std::min((int)temp1.y, dataset.images[i].image_gray.rows - 1));
			temp2.x = std::max(0, std::min((int)temp2.x, dataset.images[i].image_gray.cols - 1));
			temp2.y = std::max(0, std::min((int)temp2.y, dataset.images[i].image_gray.rows - 1));

			int t1 = (int)(dataset.images[i].image_gray(temp1));
			int t2 = (int)(dataset.images[i].image_gray(temp2));
			pixel_diff(i, j) = t1 - t2;
			//cv::Mat_<uchar> itmp = dataset.images[i].image_gray;
			//cv::circle(itmp, current_pos, 2, 120);
			//cv::circle(itmp, current_pos, param.radius[0] * dataset.images[i].bbox.width, 120);
			//cv::circle(itmp, temp1, 1, 255);
			//cv::circle(itmp, temp2, 1, 255);
			//cv::imshow("image", itmp);
			//cv::waitKey(0);
		}
	}
	/*std::ofstream fp;
	fp.open("pd1.txt");
	fp << pixel_diff;
	fp.close();*/
	return pixel_diff;
}

void randforest::Trainforest(Dataset &dataset, std::vector<Pixel_Pair> &pixel_pairs)
// pixel_diff是 500*datasetsize 大小的矩阵，每一列是一个样本的所有像素差特征
{
	cv::Mat_<int> pixel_diff;
	pixel_diff = generate_pixel_diff(dataset, pixel_pairs);
	//std::ofstream fp;
	//fp.open("pd2.txt");
	//fp << pixel_diff;
	//fp.close();
	//std::cout << "ready" << std::endl;
	srand(time(0));
	roots.resize(param.tree_num);
	//#pragma omp parallel for
	for (int num = 0; num < param.tree_num; num++)
	{
		std::vector<int> index_data, index_attr;
		index_data = Rand(dataset.datasetsize, dataset.datasetsize, true);
		index_attr = Rand(param.feat_num, (int)(param.feat_num / 3), false);

		int currentdepth = 0, leafnum = 0;
		roots[num] = Train_SingleTree(dataset, pixel_diff, index_data, index_attr, currentdepth, leafnum);
	}
	//BFS_Save_Trees("rf1.txt");
}

treenode* randforest::Train_SingleTree(Dataset &dataset, cv::Mat_<int> &pixel_diff, std::vector<int> &index_data,
	std::vector<int> &index_attr, int currentdepth, int &leafnum)
{
	if (currentdepth < param.tree_max_depth)
	{
		treenode *node = new treenode();
		std::vector<int> left_index, right_index;
		node->depth = ++currentdepth;
		if (currentdepth == this->param.tree_max_depth || index_data.empty())
		{
			node->isleaf = true;
			node->leafnum = leafnum++;
			node->lchild = NULL;
			node->rchild = NULL;
		}
		else
		{
			find_split_attr(dataset, pixel_diff, index_data, index_attr, node->attr_index, node->threshold, left_index, right_index);

			node->isleaf = false;
			node->leafnum = -1;
			node->lchild = Train_SingleTree(dataset, pixel_diff, left_index, index_attr, currentdepth, leafnum);	//构建左子树
			node->rchild = Train_SingleTree(dataset, pixel_diff, right_index, index_attr, currentdepth, leafnum);	//构建右子树
		}
		return node;
	}
	return NULL;
}

void randforest::find_split_attr(Dataset &dataset, cv::Mat_<int> &pixel_diff, std::vector<int> &index_data, std::vector<int> &index_attr, int &split_attr,
	int &thres, std::vector<int> &left_data_index, std::vector<int> &right_data_index)
{
	//cv::RNG rd(time(0));
	std::vector<int> left_index, right_index;
	double var = DBL_MAX, var_new;
	int itemp = 0;
	for (int i = 0; i < index_attr.size(); i++)
	{
		double left_var = 0, right_var = 0;
		cv::Point2f left_sum(0.0, 0.0), right_sum(0.0, 0.0);
		left_index.clear();
		right_index.clear();

		std::vector<int> datatemp;
		datatemp.reserve(index_data.size());
		for (int j = 0; j < index_data.size(); j++){
			datatemp.push_back(pixel_diff[index_data[j]][i]);
		}

		std::sort(datatemp.begin(), datatemp.end());
		//int temp_index = floor((int)(index_data.size()*(0.5 + 0.9*(rd.uniform(0.0, 1.0) - 0.5))));
		int temp_index = (int)index_data.size() / 2;
		int temp_thres = datatemp[temp_index];
		for (int j = 0; j < index_data.size(); j++)
		{
			int temp_index = index_data[j];
			if (pixel_diff[temp_index][i] <= temp_thres){
				left_index.push_back(temp_index);
				cv::Point2f dS_i = get_ith_vector(dataset.images[temp_index].dS, landmark_index);
				left_sum += dS_i;
				left_var += dS_i.dot(dS_i);
			}
			else{
				right_index.push_back(temp_index);
				cv::Point2f dS_i = get_ith_vector(dataset.images[temp_index].dS, landmark_index);
				right_sum += dS_i;
				right_var += dS_i.dot(dS_i);
			}
		}
		if (left_index.empty())
			left_var = 0;
		else{
			left_sum = left_sum / (double)(left_index.size());	//求出均值，是二维的向量
			left_var = left_var / left_index.size() - left_sum.dot(left_sum);
		}

		if (right_index.empty())
			right_var = 0;
		else{
			right_sum = right_sum / (double)(right_index.size());
			right_var = right_var / right_index.size() - right_sum.dot(right_sum);
		}

		var_new = left_var * left_index.size() + right_var * right_index.size();
		if (var_new < var)
		{
			var = var_new;
			thres = temp_thres;
			split_attr = index_attr[i];
			itemp = i;
			left_data_index = left_index;
			right_data_index = right_index;
		}
	}
	index_attr.erase(index_attr.begin() + itemp);		//用过的属性要从属性表中去掉
}


std::vector<int> randforest::Rand(int X, int x, bool flag)
{
	std::vector<int> temp(X);
	std::vector<int> index(x);
	for (int i = 0; i < X; i++)
	{
		temp[i] = i;
	}
	for (int i = 0; i < x; i++)
	{
		int ind = rand() % X;
		index[i] = temp[ind];
		if (!flag)		//无放回的（产生不重复的随机数）
		{
			//int a = temp[ind];
			temp[ind] = temp[X - 1];
			//temp[X - 1] = a;
			X--;
		}
	}
	return index;
}

void randforest::BFS_Save_Trees(const std::string &rf_file)
{
	std::ofstream fp;
	fp.open(rf_file);
	int nodenum = pow(2, param.tree_max_depth) - 1;
	for (int r_i = 0; r_i < roots.size(); r_i++)
	{
		std::vector<treenode*> nodelist;
		nodelist.reserve(nodenum);
		std::vector<std::vector<int> > childlist;
		childlist.reserve(nodenum);

		std::queue<treenode*> nodeque;
		nodeque.push(roots[r_i]);
		treenode *p;

		while (!nodeque.empty())
		{
			p = nodeque.front();
			nodeque.pop();
			nodelist.push_back(p);

			std::vector<int> tmp(2);
			if (p->lchild == NULL)
				tmp[0] = -1;
			else
			{
				tmp[0] = nodelist.size() + nodeque.size();
				nodeque.push(p->lchild);
			}

			if (p->rchild == NULL)
				tmp[1] = -1;
			else
			{
				tmp[1] = nodelist.size() + nodeque.size();
				nodeque.push(p->rchild);
			}
			childlist.push_back(tmp);
		}

		fp << nodelist.size() << ' ';		//节点总数
		for (int i = 0; i < childlist.size(); i++)
		{
			fp << childlist[i][0] << " " << childlist[i][1] << " ";
		}
		fp << std::endl;
		for (int i = 0; i < nodelist.size(); i++)
		{
			fp << nodelist[i]->threshold << ' ' << nodelist[i]->attr_index << ' '
				<< nodelist[i]->isleaf << ' ' << nodelist[i]->leafnum << ' ' << nodelist[i]->depth << ' ';
		}
		fp << std::endl;
	}
	fp.close();
}

void randforest::BFS_Load_Trees(const std::string &rf_file)
{
	std::ifstream fp;
	fp.open(rf_file);
	roots.resize(param.tree_num);
	for (int r_i = 0; r_i < param.tree_num; r_i++)
	{
		int nodenum;
		fp >> nodenum;		//读入节点数量

		std::vector<treenode*> nodelist;
		nodelist.resize(nodenum);
		std::vector<int> childlist(2);
		for (int i = 0; i < nodenum; i++)
			nodelist[i] = new treenode;
		for (int i = 0; i < nodenum; i++)
		{
			fp >> childlist[0] >> childlist[1];
			if (childlist[0] == -1)
				nodelist[i]->lchild = NULL;
			else
				nodelist[i]->lchild = nodelist[childlist[0]];

			if (childlist[0] == -1)
				nodelist[i]->rchild = NULL;
			else
				nodelist[i]->rchild = nodelist[childlist[1]];
		}
		for (int i = 0; i < nodenum; i++)
		{
			fp >> nodelist[i]->threshold >> nodelist[i]->attr_index
				>> nodelist[i]->isleaf >> nodelist[i]->leafnum >> nodelist[i]->depth;
		}
		roots[r_i] = nodelist[0];
	}
	fp.close();
}