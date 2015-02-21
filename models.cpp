#include "models.h"
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <chrono>

using namespace std;
using namespace cv;

#define USE_RANDOM_FOREST 0

void Regression(SampleInfo &sample_info, std::vector<float> &predictions, std::vector<float>& sample, std::vector<std::vector<float>>& response)
{//cadata
	vector<int> train_indices(sample_info.n_train_samples);
	iota(train_indices.begin(), train_indices.end(), 0);
	size_t n_test_samples = sample_info.n_valid_samples + sample_info.n_test_samples;

	predictions.resize(n_test_samples);
        cout<<"one"<<endl;
	Mat featsMat(sample_info.n_train_samples, sample_info.n_features, CV_32FC1, sample.data());
	Mat labsMat(sample_info.n_train_samples, 1, CV_32FC1, response[0].data());
	Mat varTypesMat(1, sample_info.n_features + 1, CV_8UC1);
	for (int i = 0; i < sample_info.n_features; i++)
	{
		if (sample_info.featTypes.empty() || sample_info.featTypes[i] == FeaturesType::Categorical)
		{
			varTypesMat.at<uchar>(0, i) = CV_VAR_CATEGORICAL;
		}
		else if (sample_info.featTypes[i] == FeaturesType::Numeric)
		{
			varTypesMat.at<uchar>(0, i) = CV_VAR_NUMERICAL;
		}
	}
	varTypesMat.at<uchar>(0, sample_info.n_features) = CV_VAR_NUMERICAL;
        cout<<"two"<<endl;

	std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::high_resolution_clock::now();

#if USE_RANDOM_FOREST
	CvRTParams rParams(4, 15, 0, false, 10, nullptr, false, 0, 1000, 0.001, CV_TERMCRIT_ITER + CV_TERMCRIT_EPS);
	CvRTrees RT;
	RT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), varTypesMat, Mat(), rParams);
#else
        cout<<"three"<<endl;
	CvGBTrees GBT;
        cout<<"four"<<endl;
	CvGBTreesParams params;
        cout<<"five"<<endl;
	params.weak_count = 11000;
	params.loss_function_type = CvGBTrees::HUBER_LOSS;
	params.shrinkage = 0.002;
	params.max_depth = 6;
	params.subsample_portion = 0.7;
        cout<<"four"<<endl;
	GBT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), varTypesMat, Mat(), params);
#endif
	long long durationTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
	cout << "Build model: " << float(durationTime) / 1000 << "sec." << endl;
	//CvRTParams params(20, 5, 0, false, 10, nullptr, false, 0, 1000, 0, 0);
	////params.max_depth = 10;
	//params.nactive_vars = 0;// sample_info.n_features;
	////params.term_crit = 

	//CvRTrees ERT;
	//ERT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), Mat(), Mat(), params);
	for (int i = 0; i < sample_info.n_valid_samples + sample_info.n_test_samples; i++)
	{
		Mat sampleMat(1, sample_info.n_features, CV_32FC1, sample.data() + sample_info.n_features*(sample_info.n_train_samples + i));
		//predictions[i] = GBT.predict(sampleMat);
		//predictions[i] = ERT.predict(sampleMat);
#if USE_RANDOM_FOREST
		predictions[i] = RT.predict(sampleMat);
#else
		predictions[i] = GBT.predict(sampleMat);
#endif
	}
}

void ClassificationMART(SampleInfo &sample_info, std::vector<float> &class_probs, std::vector<float> &predictions, 
	std::vector<float>& sample, std::vector<std::vector<float>>& response, vector<int>& labelsMap,
	string basename)
{//digits, dorothea, newsgroups
	size_t n_test_samples = sample_info.n_valid_samples + sample_info.n_test_samples;

	class_probs.resize(sample_info.n_response_classes * n_test_samples);
	predictions.resize(n_test_samples);

	Mat featsMat(sample_info.n_train_samples, sample_info.n_features, CV_32FC1, sample.data());
	Mat labsMat(sample_info.n_train_samples, 1, CV_32SC1, response[0].data());
	Mat varTypesMat(1, sample_info.n_features + 1, CV_8UC1);
	bool newsgroups = basename == "newsgroups";
	for (int i = 0; i < sample_info.n_features; i++)
	{
		if (sample_info.featTypes.empty() || sample_info.featTypes[i] == FeaturesType::Categorical || sample_info.featTypes[i] == FeaturesType::Binary || newsgroups)
		{
			varTypesMat.at<uchar>(0, i) = CV_VAR_CATEGORICAL;
		}
		else if (sample_info.featTypes[i] == FeaturesType::Numeric)
		{
			varTypesMat.at<uchar>(0, i) = CV_VAR_NUMERICAL;
		}
	}
	varTypesMat.at<uchar>(0, sample_info.n_features) = CV_VAR_CATEGORICAL;
        

	std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::high_resolution_clock::now();
#if USE_RANDOM_FOREST
	size_t nTrees = 1000;
	CvRTParams rParams(4, 5, 0, false, 10, nullptr, false, 0, nTrees, 0.01, CV_TERMCRIT_ITER + CV_TERMCRIT_EPS);
	CvRTrees RT;
	cout << "begin training" << endl;
	RT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), varTypesMat, Mat(), rParams);
#else
	CvGBTrees GBT;
	CvGBTreesParams params;
	if (basename == "dorothea")
	{
		params.weak_count = 50;
		params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
		params.shrinkage = 0.1;
		params.max_depth = 4;
		params.subsample_portion = 0.7;
	}
	else if (basename == "newsgroups")
	{
		params.weak_count = 1;
		params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
		params.shrinkage = 1;
		params.max_depth = 3;
		params.subsample_portion = 0.6;
	}
	// else if (basename == "digits")
	// {
        //     params.weak_count = 1;//10;
	// 	params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
	// 	params.shrinkage = 0.1;
	// 	params.max_depth = 4;
	// 	params.subsample_portion = 0.7;
	// 	//params.weak_count = 10;
	// 	//params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
	// 	//params.shrinkage = 0.1;
	// 	//params.max_depth = 4;
	// 	//params.subsample_portion = 0.7;
	// 	//cout << "GBT for digits\n";
	// }
	else
	{
            long complexity = (sample_info.n_features + sqrt((double)sample_info.n_features))*
                sample_info.n_response_classes * sample_info.n_train_samples;
            cout<<"complexity = "<<complexity<<endl;
            params.weak_count = double(long(15000000) *sample_info.time_budget*(0.5))/complexity + 1;
            params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
            params.shrinkage = min(0.1, double(20)/params.weak_count);
            params.max_depth = 4;
            params.subsample_portion = 0.6;
	}

	cout << "GBT for " + basename << endl;
	cout << endl <<"params:"<<endl
		<< "weak_count = " << params.weak_count<<endl
		<< "shrinkage = " << params.shrinkage << endl
		<< "max_depth = " << params.max_depth << endl
		<< "params.subsample_portion = " << params.subsample_portion << endl
		<< endl;
	GBT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), varTypesMat, Mat(), params);
#endif
	long long durationTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
	cout << "Build model: " << float(durationTime) / 1000 << "sec." << endl;
	int class_idx = response[0][0] == 0;
	vector<float> resp(sample_info.n_response_classes);
	for (int i = 0; i < sample_info.n_valid_samples + sample_info.n_test_samples; i++)
	{
		Mat sampleMat(1, sample_info.n_features, CV_32FC1, sample.data() + sample_info.n_features*(sample_info.n_train_samples + i));
#if USE_RANDOM_FOREST
		for (auto & it : resp) it = 0.f;
		for (int t = 0; t < nTrees; t++)
		{
			size_t predVal = RT.get_tree(t)->predict(sampleMat)->class_idx;
			resp[predVal] = resp[predVal] + 1;
		}
		for (int j = 0; j < sample_info.n_response_classes; j++)
		{
			class_probs[sample_info.n_response_classes * i + j] = resp[j] / nTrees;
		}
#else
		for (int j = 0; j < sample_info.n_response_classes; j++)
		{
			double temp = GBT.predict(sampleMat, cv::Mat(), cv::Range::all(), j);
			double prob = 1. / (1. + exp(-2.*temp));
			class_probs[sample_info.n_response_classes * i + labelsMap[j]] = prob;
		}

		//double temp = GBT.predict(sampleMat, cv::Mat(), cv::Range::all(), 1 - class_idx);
		//double prob = 1. / (1. + exp(-2.*temp));
		//class_probs[sample_info.n_response_classes * i] =  prob;
		//class_probs[sample_info.n_response_classes * i + 1] = 1 - prob;
		//resp[1] = 1 - prob;
		//auto minEl = std::min_element(resp.begin(),resp.end());
		//auto sumEl = std::accumulate(resp.begin(), resp.end(), -(*minEl) * sample_info.n_response_classes);
		//if (basename == "dorothea")
		//{
		//	class_probs[sample_info.n_response_classes * i] = resp[0];
		//	class_probs[sample_info.n_response_classes * i + 1] = resp[1];
		//}
		//else
		//{
		//	for (int j = 0; j < sample_info.n_response_classes; j++)
		//	{
		//		//class_probs[sample_info.n_response_classes * i + labelsMap[j]] = (resp[j] - *minEl) / sumEl;
		//		//class_probs[sample_info.n_response_classes * i + labelsMap[j]] = resp[j];
		//		//class_probs[sample_info.n_response_classes * i + j] = resp[j];
		//		class_probs[sample_info.n_response_classes * i + labelsMap[j]] = resp[j];
		//	}
		//}
#endif
	}
	cout << "class_probs[0] = " << class_probs[0] << endl;
}

void MultilabelClassificationMART(SampleInfo &sample_info, std::vector<float> &class_probs, 
	std::vector<float> &predictions, std::vector<float>& sample, std::vector<std::vector<float>>& response)
{// adult
	size_t n_test_samples = sample_info.n_valid_samples + sample_info.n_test_samples;

	class_probs.resize(2 * sample_info.n_response_classes * n_test_samples);
	predictions.resize(n_test_samples);
        cv::theRNG().state = 1;//time(NULL);
	Mat missingMat;
	vector<float> missing;
	if (sample_info.is_missing)
	{
		missing.resize(sample.size());
		for (int i = 0; i < sample.size();i++)
		{
			missing[i] = (sample[i] == MISSED_VALUE_FLOAT);
			if (missing[i])
			{
				sample[i] = 0;
			}
		}
		missingMat = Mat(sample_info.n_train_samples, sample_info.n_features, CV_8UC1, missing.data());
	}
	Mat featsMat(sample_info.n_train_samples, sample_info.n_features, CV_32FC1, sample.data());
	Mat varTypesMat(1, sample_info.n_features+1, CV_8UC1);
	for (int i = 0; i < sample_info.n_features; i++)
	{
		if (sample_info.featTypes.empty() || sample_info.featTypes[i] == FeaturesType::Categorical)
		{
			varTypesMat.at<uchar>(0, i) = CV_VAR_CATEGORICAL;
		}
		else if (sample_info.featTypes[i] == FeaturesType::Numeric)
		{
			varTypesMat.at<uchar>(0, i) = CV_VAR_NUMERICAL;
		}
	}
	varTypesMat.at<uchar>(0, sample_info.n_features) = CV_VAR_CATEGORICAL;

#if USE_RANDOM_FOREST
	size_t nTrees = 1000;
	CvRTParams rParams(7, 20, 0, false, 20, nullptr, false, 0, nTrees, 0.001, CV_TERMCRIT_ITER + CV_TERMCRIT_EPS);
	CvRTrees RT;
#else
	CvGBTrees GBT;
	CvGBTreesParams params;
	params.weak_count = 500;// 6000;
	params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
	params.shrinkage = 0.05;
	params.max_depth = 6;
	params.subsample_portion = 0.7;
	//params.use_surrogates = true;
	cout << "GBT for multilabel " << endl;
	cout << endl <<"params:"<<endl
		<< "weak_count = " << params.weak_count<<endl
		<< "shrinkage = " << params.shrinkage << endl
		<< "max_depth = " << params.max_depth << endl
             << "params.subsample_portion = " << params.subsample_portion << endl;
#endif
	std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::high_resolution_clock::now();
	long long durationTime = 0;

	for (size_t c = 0; c < sample_info.n_response_classes; ++c)
	{
		cout << "c = " << c << endl;
		Mat labsMat(sample_info.n_train_samples, 1, CV_32SC1, response[c].data());
#if USE_RANDOM_FOREST
		RT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), varTypesMat, missingMat, rParams);
		cout<<RT.getVarImportance();
#else
		GBT.train(featsMat, CV_ROW_SAMPLE, labsMat, Mat(), Mat(), varTypesMat, missingMat, params);
#endif
		int classIdx = response[c][0] == 0;

		for (int i = 0; i < sample_info.n_valid_samples + sample_info.n_test_samples; i++)
		{

			float* myPtr = missing.data() + sample_info.n_features*(sample_info.n_train_samples + i);
			Mat missingMatSample = Mat(1, sample_info.n_features, CV_8UC1, myPtr);
			Mat sampleMat(1, sample_info.n_features, CV_32FC1, sample.data() + sample_info.n_features*(sample_info.n_train_samples + i));

#if USE_RANDOM_FOREST
			double prob = RT.predict_prob(sampleMat, missingMatSample);
			class_probs[2 * c * n_test_samples + 2 * i + 1] = prob;
			class_probs[2 * c * n_test_samples + 2 * i ] = 1 - prob;
#else
			double temp0 = GBT.predict(sampleMat, missingMatSample, cv::Range::all(), 1 - classIdx);
			double prob0 = 1. / (1. + exp(-2.*temp0));
			class_probs[2 * c * n_test_samples + 2 * i] = prob0;
			class_probs[2 * c * n_test_samples + 2 * i + 1] = 1 - prob0;
#endif
		}
		durationTime += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
		cout << "time spent: " << float(durationTime) / 1000 << "sec." << endl;
	}
	
	
}
