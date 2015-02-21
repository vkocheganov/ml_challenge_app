#include "properties.h"
#include <vector>
#include <float.h>
const float MISSED_VALUE_FLOAT = FLT_MAX;

//
//void MultilabelClassification(SampleInfo &sample_info, std::vector<float> &class_probs,
//	std::vector<float> &predictions, CClassifierSampleWrapper &sample);
//
void MultilabelClassificationMART(SampleInfo &sample_info, std::vector<float> &class_probs,
	std::vector<float> &predictions, std::vector<float>& sample, std::vector<std::vector<float>>& response);

//void Classification(SampleInfo &sample_info, std::vector<float> &class_probs,
//	std::vector<float> &predictions, CClassifierSampleWrapper &sample);

void Regression(SampleInfo &sample_info, std::vector<float> &predictions,
	std::vector<float>& sample, std::vector<std::vector<float>>& response);


void ClassificationMART(SampleInfo &sample_info, std::vector<float> &class_probs,
	std::vector<float> &predictions, std::vector<float>& sample, std::vector<std::vector<float>>& response,
	std::vector<int>& labelsMap, std::string basename);
