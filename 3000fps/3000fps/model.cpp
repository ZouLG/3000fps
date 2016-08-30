//#include "model.h"
#include "headers.h"
#include <direct.h>
#include <ctime>
#include <io.h>

void Regressor::generate_pixel_pairs()
{
	cv::RNG rd(time(0));
	pixel_pair.resize(param.landmark_num);
	for (int i = 0; i < param.landmark_num; i++)
	{
		pixel_pair[i].reserve(param.feat_num);
		for (int j = 0; j < param.feat_num; j++)
		{
			float th1 = rd.uniform(0.0, 2 * CV_PI), th2 = rd.uniform(0.0, 2 * CV_PI);
			float r1 = rd.uniform(0.0, param.radius[stage]), r2 = rd.uniform(0.0, param.radius[stage]);

			Pair temp;
			temp.pixel1 = cv::Point2f(r1*cos(th1), r1*sin(th1));
			temp.pixel2 = cv::Point2f(r2*cos(th2), r2*sin(th2));
			pixel_pair[i].push_back(temp);
		}
	}
}

void Regressor::Get_LBF(Image &image, feature_node *x)
{
	int index = 1, dim = pow(2, param.tree_max_depth - 1);
	int k = 0;
	for (int i = 0; i < param.landmark_num; i++)
	{
		for (int j = 0; j < param.tree_num; j++)
		{
			treenode *p = rf[i].roots[j];
			while (!(p->isleaf))
			{
				cv::Point2f pixel1 = pixel_pair[i][p->attr_index].pixel1;
				cv::Point2f pixel2 = pixel_pair[i][p->attr_index].pixel2;

				cv::Mat_<double> tmp = image.affine_mat * cv::Mat_<double>(pixel1);
				pixel1 = cv::Point2f(tmp);
				tmp = image.affine_mat * cv::Mat_<double>(pixel2);
				pixel2 = cv::Point2f(tmp);

				cv::Point2f current_pos = get_ith_vector(image.current_shape, i);
				pixel1 += current_pos;
				pixel2 += current_pos;

				pixel1.x = std::max(0, std::min((int)pixel1.x, image.image_gray.cols - 1));
				pixel1.y = std::max(0, std::min((int)pixel1.y, image.image_gray.rows - 1));
				pixel2.x = std::max(0, std::min((int)pixel2.x, image.image_gray.cols - 1));
				pixel2.y = std::max(0, std::min((int)pixel2.y, image.image_gray.rows - 1));

				int t1 = (int)(image.image_gray(pixel1));
				int t2 = (int)(image.image_gray(pixel2));
				int p_d = t1 - t2;

				if (p_d <= p->threshold)
					p = p->lchild;
				else
					p = p->rchild;
			}
			x[k].index = index + p->leafnum;
			x[k].value = 1.0;
			k++;
			index += dim;
		}
	}
	x[k].index = -1;
	x[k].value = -1.0;
}

void Regressor::Save_Regressor(const std::string DirectoryPath)
{
	std::string Regress_Path = DirectoryPath + "/Regressor_";
	char tmp[3] = "";
	_itoa(stage, tmp, 10);
	std::string temp = tmp;
	Regress_Path += temp;
	if (_access(Regress_Path.c_str(),0) == -1)			//´´½¨Ä¿Â¼
		_mkdir(Regress_Path.c_str());

	std::string file_path = Regress_Path + "/Feature_Location.txt";
	std::ofstream fp;

	fp.open(file_path);
	for (int j = 0; j < param.feat_num; j++)
	{
		for (int i = 0; i < param.landmark_num; i++)
			fp << pixel_pair[i][j].pixel1.x << ' ' << pixel_pair[i][j].pixel1.y << ' '
				<< pixel_pair[i][j].pixel2.x << ' ' << pixel_pair[i][j].pixel2.y << ' ';
		fp << std::endl;
	}
	fp.close();
	
	for (int i = 0; i < param.landmark_num; i++)
	{
		_itoa(i, tmp, 10);
		temp = tmp;
		temp += ".txt";
		file_path = Regress_Path + "/Randforest_" + temp;
		rf[i].BFS_Save_Trees(file_path);
		file_path = Regress_Path + "/Model_x_" + temp;
		save_model(file_path.c_str(), Model_x[i]);
		file_path = Regress_Path + "/Model_y_" + temp;
		save_model(file_path.c_str(), Model_y[i]);
	}
}

void Regressor::Load_Regressor(const std::string RegressorDirectory)
{
	pixel_pair.resize(param.landmark_num);
	std::string filepath = RegressorDirectory + "/Feature_Location.txt";
	std::ifstream fp;
	fp.open(filepath);
	for (int i = 0; i < param.landmark_num; i++)
		pixel_pair[i].resize(param.feat_num);
	for (int j = 0; j < param.feat_num; j++)
	{
		for (int i = 0; i < param.landmark_num; i++)
			fp >> pixel_pair[i][j].pixel1.x >> pixel_pair[i][j].pixel1.y >>
			pixel_pair[i][j].pixel2.x >> pixel_pair[i][j].pixel2.y;

	}
	fp.close();

	rf.resize(param.landmark_num);
	Model_x.resize(param.landmark_num);
	Model_y.resize(param.landmark_num);
	for (int i = 0; i < param.landmark_num; i++)
	{
		char tmp[3] = "";
		_itoa(i, tmp, 10);
		std::string temp = tmp;
		temp += ".txt";
		filepath = RegressorDirectory + "/Randforest_" + temp;
		rf[i].param = param;
		rf[i].landmark_index = i;
		rf[i].BFS_Load_Trees(filepath);
		filepath = RegressorDirectory + "/Model_x_" + temp;
		Model_x[i] = load_model(filepath.c_str());
		filepath = RegressorDirectory + "/Model_y_" + temp;
		Model_y[i] = load_model(filepath.c_str());
	}
}

void Model::Train(Dataset &dataset)
{
	current_stage = 0;
	meanshape = dataset.meanshape.clone();
	struct problem* prob = new struct problem;
	prob->l = dataset.datasetsize;
	prob->bias = -1;
	prob->n = param.landmark_num * param.tree_num * pow(2, param.tree_max_depth - 1);

	struct parameter* regression_param = new struct parameter;
	regression_param->solver_type = L2R_L2LOSS_SVR_DUAL;
	regression_param->C = 1.0 / dataset.datasetsize;
	regression_param->p = 0;
	regression_param->eps = 0.00001;

	struct feature_node **LBF = new feature_node*[dataset.datasetsize];
	for (int i = 0; i < dataset.datasetsize; i++){
		LBF[i] = new feature_node[param.landmark_num * param.tree_num + 1];
	}
	prob->x = LBF;

	double *l_x = new double[dataset.datasetsize];
	double *l_y = new double[dataset.datasetsize];

	regress.resize(param.stage_num);
	std::cout << "Start training regressors, " << param.stage_num << " stages in total." << std::endl;
	for (; current_stage < param.stage_num; current_stage++)
	{
		std::cout << std::endl;
		std::cout << "Current stage is " << current_stage << std::endl;
		regress[current_stage].param = param;
		regress[current_stage].stage = current_stage;
		regress[current_stage].generate_pixel_pairs();
		regress[current_stage].rf.resize(param.landmark_num);
		regress[current_stage].Model_x.resize(param.landmark_num);
		regress[current_stage].Model_y.resize(param.landmark_num);

		std::cout << "Training all the random forests..." << std::endl;
		//#pragma omp parallel for
		for (int j = 0; j < param.landmark_num; j++){
			regress[current_stage].rf[j].param = param;
			regress[current_stage].rf[j].landmark_index = j;
			regress[current_stage].rf[j].Trainforest(dataset, regress[current_stage].pixel_pair[j]);
			//char tmp[3] = "";
			//_itoa(j,tmp,10);
			//std::string path = tmp;
			//path += "_forest.txt";
			//regress[i].rf[j].BFS_Save_Trees(path);
		}

		std::cout << "Getting the LBF code of all training data..." << std::endl;
		for (int j = 0; j < dataset.datasetsize; j++){
			regress[current_stage].Get_LBF(dataset.images[j], LBF[j]);
		}

		std::cout << "Doing global regression..." << std::endl;
		for (int i = 0; i < param.landmark_num; i++)
		{
			for (int j = 0; j < dataset.datasetsize; j++)
			{
				l_x[j] = dataset.images[j].dS(i, 0);
				l_y[j] = dataset.images[j].dS(i, 1);
			}
			prob->y = l_x;
			regress[current_stage].Model_x[i] = train(prob, regression_param);

			prob->y = l_y;
			regress[current_stage].Model_y[i] = train(prob, regression_param);
		}

		std::cout << "Updating current shapes..." << std::endl;
		for (int i = 0; i < dataset.datasetsize; i++)
		{
			Image &imtmp = dataset.images[i];
			for (int j = 0; j < param.landmark_num; j++)
			{
				imtmp.dS(j, 0) = predict(regress[current_stage].Model_x[j], LBF[i]);
				imtmp.dS(j, 1) = predict(regress[current_stage].Model_y[j], LBF[i]);
			}

			//imtmp.shape_inter.push_back(imtmp.current_shape);

			cv::Mat_<double> shape_ = imtmp.dS + Center_and_scale(imtmp.bbox, imtmp.current_shape);
			imtmp.current_shape = Center_and_scale(shape_, imtmp.bbox);
			Round_shape(imtmp.image_gray, imtmp.current_shape);
			imtmp.dS = Center_and_scale(imtmp.bbox, imtmp.shape) - Center_and_scale(imtmp.bbox, imtmp.current_shape);
		}
		dataset.Cal_Error();

	}
	for (int i = 0; i < dataset.datasetsize; i++){
		delete[]LBF[i];
	}
	delete[]LBF;
	delete[]l_x;
	delete[]l_y;
}

void Model::Test(Image &image)
{
	std::vector<cv::Rect> bboxs;
	//clock_t t1 = clock();
	facedetect(image.image_gray, bboxs);
	int dim = pow(2, param.tree_max_depth - 1);
	for (int i = 0; i < bboxs.size(); i++)
	{
		image.bbox = bboxs[i];
		image.current_shape = Center_and_scale(meanshape, image.bbox);
		image.affine_mat = Get_Affine_Mat(meanshape, image.current_shape);
		//std::cout << image.affine_mat << std::endl;
		feature_node* x = new feature_node[param.landmark_num * param.tree_num + 1];
		for (current_stage = 0; current_stage < param.stage_num; current_stage++)
		{
			regress[current_stage].Get_LBF(image, x);

			cv::Mat_<double> dS_tmp(param.landmark_num, 2);
			for (int l_i = 0; l_i < param.landmark_num; l_i++)
			{
				dS_tmp(l_i, 0) = predict(regress[current_stage].Model_x[l_i], x);
				dS_tmp(l_i, 1) = predict(regress[current_stage].Model_y[l_i], x);
			}

			cv::Mat_<double> shape_ = dS_tmp + Center_and_scale(image.bbox, image.current_shape);
			image.current_shape = Center_and_scale(shape_, image.bbox);
			Round_shape(image.image_gray, image.current_shape);
			image.affine_mat = Get_Affine_Mat(meanshape, image.current_shape);
			//Draw_shapes(image.image_gray, image.current_shape);
			//cv::waitKey(0);
		}
		Draw_shapes(image.image_gray, image.current_shape);
		cv::waitKey(0);
		delete[]x;
	}
	//clock_t t2 = clock();
	//std::cout << t2 - t1 << std::endl;
}

void Model::TestCamera()
{
	double scale = 2;
	cv::VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		std::cerr << "Camera open failed..." << std::endl;
		return;
	}

	Image im = Image();
	cv::Mat temp;
	cv::Mat_<double> meanshape_ = meanshape.clone();

	feature_node* x = new feature_node[param.landmark_num * param.tree_num + 1];
	while (1)
	{
		std::vector<cv::Rect> bboxs;
		cap.read(temp);
		cv::cvtColor(temp, im.image_gray, CV_RGB2GRAY);
		cv::resize(im.image_gray, temp, cv::Size(), 1 / scale, 1 / scale, CV_INTER_LINEAR);
		facedetect(temp, bboxs);
		if (bboxs.empty())
		{
			cv::namedWindow("window 1", cv::WINDOW_AUTOSIZE);
			cv::imshow("window 1", im.image_gray);
			if (27 == cv::waitKey(15))
				return;
			continue;
		}
		im.bbox.x = bboxs[0].x * scale; im.bbox.y = bboxs[0].y * scale;
		im.bbox.width = bboxs[0].width * scale; im.bbox.height = bboxs[0].height * scale;

		im.current_shape = Center_and_scale(meanshape_, im.bbox);
		im.affine_mat = Get_Affine_Mat(meanshape, im.current_shape);
		for (current_stage = 0; current_stage < param.stage_num; current_stage++)
		{
			regress[current_stage].Get_LBF(im, x);

			cv::Mat_<double> dS_tmp(param.landmark_num, 2);
#pragma omp parallel for
			for (int l_i = 0; l_i < param.landmark_num; l_i++)
			{
				dS_tmp(l_i, 0) = predict(regress[current_stage].Model_x[l_i], x);
				dS_tmp(l_i, 1) = predict(regress[current_stage].Model_y[l_i], x);
			}

			cv::Mat_<double> shape_ = dS_tmp + Center_and_scale(im.bbox, im.current_shape);
			im.current_shape = Center_and_scale(shape_, im.bbox);
			Round_shape(im.image_gray, im.current_shape);
			im.affine_mat = Get_Affine_Mat(meanshape, im.current_shape);
		}
		Draw_shapes(im.image_gray, im.current_shape, im.bbox);
		//meanshape_ = Center_and_scale(im.bbox, im.current_shape);
		cv::waitKey(1);
	}
	delete[]x;
	cv::destroyAllWindows();
}


void Model::Save_Model(const std::string Model_Path)
{
	std::cout << "Start saving model..." << std::endl;
	std::string model_path = Model_Path + "/Model";
	if (_access(model_path.c_str(), 0) == -1)
		_mkdir(model_path.c_str());

	std::ofstream fp;
	char tmp;
	fp.open(model_path + "/Param_and_Meanshape.txt");
	fp << param.landmark_num << std::endl;
	fp << param.stage_num << std::endl;
	for (int i = 0; i < param.stage_num; i++)
		fp << param.radius[i] << " ";
	fp << std::endl;
	fp << param.feat_num << std::endl;
	fp << param.tree_num << std::endl;
	fp << param.tree_max_depth << std::endl;

	for (int i = 0; i < param.landmark_num; i++)
		fp << meanshape(i, 0) << " " << meanshape(i, 1) << std::endl;

	fp.close();
	for (int i = 0; i < regress.size(); i++)
	{
		regress[i].Save_Regressor(model_path);
	}

}

void Model::Load_Model(const std::string Model_path)
{
	std::cout << "loading model..." << std::endl;
	std::ifstream fp;
	fp.open(Model_path + "/Param_and_Meanshape.txt");
	fp >> param.landmark_num;
	fp >> param.stage_num;
	param.radius.resize(param.stage_num);
	for (int i = 0; i < param.stage_num; i++)
		fp >> param.radius[i];
	fp >> param.feat_num;
	fp >> param.tree_num;
	fp >> param.tree_max_depth;

	meanshape = cv::Mat_<double>(param.landmark_num, 2, 0.0);
	for (int i = 0; i < param.landmark_num; i++)
		fp >> meanshape(i, 0) >> meanshape(i, 1);
	fp.close();

	std::string RegressorDirectory;
	regress.resize(param.stage_num);
	for (int i = 0; i < param.stage_num; i++)
	{
		char temp[3] = "";
		_itoa(i, temp, 10);
		std::string tmp = temp;
		RegressorDirectory = Model_path + "/Regressor_" + tmp;
		regress[i].param = param;
		regress[i].stage = i;
		regress[i].Load_Regressor(RegressorDirectory);
	}
	
	std::cout << "loading completed." << std::endl;
}