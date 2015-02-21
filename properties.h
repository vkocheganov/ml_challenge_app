#pragma once
#include <map>
#include<vector>
#include<string>
using namespace std;
typedef std::map<std::string, std::string> properties_map_t;

enum class FeaturesType
{
	Categorical,
	Numeric,
	Mixed,
	Binary
};


struct SampleInfo
{
	std::vector<FeaturesType> featTypes;
	std::string name;
	bool is_multilabel;
	bool is_sparse;
	int n_response_classes;
	FeaturesType features_type;
	size_t n_features;
	size_t n_train_samples;
	size_t n_valid_samples;
	size_t n_test_samples;
	size_t time_budget;
	bool is_missing;

	std::string Dump() const;
};

properties_map_t GetPropertiesMap(const std::string& filename);
SampleInfo GetSampleInfo(properties_map_t& properties_map);
std::vector<FeaturesType> GetFeaturesInfo(std::string fileName);
