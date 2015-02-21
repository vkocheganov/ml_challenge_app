#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <chrono>

using namespace cv;
using namespace std;

#include "properties.h"
#include "io.h"
#include "models.h"
#include <numeric>
#if 0

int main()
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	const size_t N = 1000;
	const size_t dim = 2;
	vector<float> feats(N*dim);
	feats[0] = feats[1] = 0;
	for (int i = 2; i < N*dim; i++)
	{
		feats[i] = rand() % 512;
	}
	Mat featsMat(N, dim, CV_32FC1, feats.data());

	vector<float> labs(N);
	for (int i = 0; i < N; i++)
	{
		labs[i] = (feats[2 * i] < 256) + (feats[2 * i+ 1] < 256);
		//labs[i] = (feats[2 * i] < 256) ;
	}
	Mat labsMat(N, 1, CV_32FC1, labs.data());
	// Set up SVM's parameters
	// Train

	//CvDTreeParams dtParams;
	//dtParams.max_depth = 10;
	//dtParams.min_sample_count = 

	const size_t layers = 3;
	vector<int> layerSizes = {(int)dim,5,1};

	Mat layerSizesMat(layers, 1, CV_32SC1, layerSizes.data());

	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.01f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	//params.bp_dw_scale = 0.05f;
	//params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	CvANN_MLP AN;
	AN.create(layerSizesMat);
	AN.train(featsMat, labsMat, cv::Mat(), cv::Mat(), params);

	//CvDTree DT;
	//DT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat());
	//CvGBTreesParams params;
	//params.weak_count = 10;
	//params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
	//params.shrinkage = 0.025;
	//params.max_depth = 3;
	//params.subsample_portion = 0.6;

	//size_t nTrees = 10;
	//CvRTParams rParams(7, 20, 0, false, 10, nullptr, true, 0, nTrees, 0.001, CV_TERMCRIT_ITER + CV_TERMCRIT_EPS);
	////rParams.min_sample_count
	////rParams.nactive_vars = 2;
	//CvRTrees RT;
	//RT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), Mat(), Mat(), rParams);

	//CvERTrees ERT;
	//ERT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), Mat(), Mat(), rParams);

	//CvGBTrees GBT;
	//GBT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), Mat(), Mat(), params);

	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0,0,255);
	//cout << "tree count = " << RT.get_tree_count() << endl;
	vector<double> resp(3);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{

		vector<float> temp = { (float)j, (float)i };
		Mat sampleMat(1, 2, CV_32FC1, temp.data());// = (Mat_<float>(1, 2) << j, i);
		//double response0 = GBT.predict(sampleMat, cv::Mat(), cv::Range::all(), 0);
		//double response1 = GBT.predict(sampleMat, cv::Mat(), cv::Range::all(), 1);
		//double response2 = GBT.predict(sampleMat, cv::Mat(), cv::Range::all(), 2);

		//double response0 = ERT.predict(sampleMat);
		//double response1 = ERT.predict(sampleMat);
		//double response2 = ERT.predict(sampleMat);

		//int classresp = response0 > response1 ? (response0 > response2 ? 0 : 2) :
		//	(response1 > response2 ? 1 : 2);
		//int classresp = ERT.predict(sampleMat);
//		int classresp = DT.predict(sampleMat)->class_idx;
		Mat outputs(1, 1, CV_32SC1);
		AN.predict(sampleMat, outputs);
		int classresp = outputs.at<float>(0, 0);
		//for (auto& it : resp) it = 0.f;
		//for (int t = 0; t < RT.get_tree_count(); t++)
		//{
		//	size_t pred = RT.get_tree(t)->predict(sampleMat)->class_idx;
		//	if (pred >= 3)
		//	{
		//		cout << "pred = " << pred << endl;
		//		__debugbreak();
		//	}
		//	resp[pred]++;
		//}

		//double classresp = RT.predict(sampleMat);
		if (classresp == 0)
			image.at<Vec3b>(i, j) = green;
		else if (classresp == 1)
			image.at<Vec3b>(i, j) = blue;
		else
			image.at<Vec3b>(i, j) = red;
		}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

	imwrite("result.png", image);        // save the image

	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);

}

#else



int main(int argc, char *argv[])
{
	if (argc != 6)
	{
		std::cout << "Usage: AutoML.exe input_dir output_dir base_file_name debug_level max_budget\n";
		return 1;
	}
	//setNumThreads(2);
	string input_dir = argv[1];
	string output_dir = argv[2];
	string base_name = argv[3];
	int debug_level = atoi(argv[4]);
	int max_budget = atoi(argv[5]);

	cout << endl<<"input_dir: " << input_dir << endl <<
		"output_dir: " << output_dir << endl <<
		"base_name : " << base_name << endl <<
		"debug_level : " << debug_level << endl <<
		"max_budget : " << max_budget << endl;

	// read data set properties
	string properties_file = input_dir + "/" + base_name + "/" + base_name + "_public.info";
	string featType_file = input_dir + "/" + base_name + "/" + base_name + "_feat.type";
	properties_map_t ds_properties = GetPropertiesMap(properties_file);
#if 0	
	for (auto it : ds_properties)
	{
		cout << it.first << ": " << it.second << endl;
	}
#endif

        cout << "Getsample Info: "<<endl;
	SampleInfo sample_info = GetSampleInfo(ds_properties);
        cout << "GetFeatures Info: "<<endl;
	sample_info.featTypes = GetFeaturesInfo(featType_file);
        cout << "GetFeatures Info: "<<endl;
	cout << sample_info.Dump();

	// create sample
	vector<float> sample;
	vector<vector<float>> response;
	std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::high_resolution_clock::now();
	vector<int> labelsMap;
        cout << "CreateSample start: "<<endl;
	CreateSample(input_dir, sample_info, sample, response, labelsMap);
	long long durationTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
	cout << "CreateSample: " << float(durationTime) / 1000 << "sec." << endl;

	std::vector<float> predictions;
	std::vector<float> class_probs;

	//sample.Dump(output_dir + "\\" + base_name + "_sample.csv");

	//cv::theRNG().state = 0;
	startTime = std::chrono::high_resolution_clock::now();
	if (sample_info.is_multilabel)
	{
		MultilabelClassificationMART(sample_info, class_probs, predictions, sample, response);
	}
	else if (sample_info.n_response_classes >= 2)
	{
		ClassificationMART(sample_info, class_probs, predictions, sample, response, labelsMap, base_name);
	}
	else
	{
		Regression(sample_info, predictions, sample, response);
	}
	durationTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
	
	startTime = std::chrono::high_resolution_clock::now();
	DumpResults(predictions, class_probs, output_dir, sample_info);
	durationTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
	return 0;
}
#endif
