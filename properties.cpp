#include "properties.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

string unquote(const string str)
{
	auto start = str.begin();
	auto finish = str.end();

	while (start < finish && (isspace(*start) || *start == '\''))
		++start;

	for (; finish > start; --finish)
	{
		char tmp = *(finish - 1);
		if (!isspace(tmp) && tmp != '\'')
			break;
	}

	return string(start, finish);
}

properties_map_t GetPropertiesMap(const std::string& filename)
{
	properties_map_t result;
	ifstream ifs(filename);
	

	while (ifs)
	{
		string name, stub, value;
		ifs >> name >> stub;
		getline(ifs, value);
		value = unquote(value);
		if (!value.empty())
			result[name] = value;
	}
	return result;
}

vector<FeaturesType> GetFeaturesInfo(string fileName)
{
	vector<FeaturesType> tempVec;
	ifstream ifs(fileName);
	if (ifs.is_open())
	{
		while (ifs)
		{
			string name;
			ifs >> name;
			if (name == "Categorical")
				tempVec.push_back(FeaturesType::Categorical);
			else if (name == "Numerical")
				tempVec.push_back(FeaturesType::Numeric);
			else if (name == "Binary")
				tempVec.push_back(FeaturesType::Binary);
			else if (name != "")
                            exit(1);
		}
	}
	return tempVec;
}

SampleInfo GetSampleInfo(properties_map_t& properties_map)
{
	SampleInfo result;
	result.name = properties_map["name"];
	result.is_multilabel = properties_map["task"] == "multilabel.classification";
	result.n_features = (size_t)atoi(properties_map["feat_num"].c_str());
	result.n_train_samples = (size_t)atoi(properties_map["train_num"].c_str());
	result.n_test_samples = (size_t)atoi(properties_map["test_num"].c_str());
	result.n_valid_samples = (size_t)atoi(properties_map["valid_num"].c_str());
	result.n_response_classes = (size_t)atoi(properties_map["label_num"].c_str());
	result.time_budget = (size_t)atoi(properties_map["time_budget"].c_str());
	result.is_sparse = (bool)atoi(properties_map["is_sparse"].c_str());
	result.is_missing= (bool)atoi(properties_map["has_missing"].c_str());

        cout<<"hello"<<endl;
	char ft = properties_map["feat_type"][0];
	switch (ft)
	{
	case 'N': result.features_type = FeaturesType::Numeric; break;
	case 'B': result.features_type = FeaturesType::Binary; break;
	case 'C': result.features_type = FeaturesType::Categorical; break;
	case 'M': result.features_type = FeaturesType::Mixed; break;
	default: cout<<properties_map["feat_type"] <<endl; exit(1);
	}
        cout<<"hello"<<endl;

	return result;
}

string SampleInfo::Dump() const
{
	ostringstream os;
	os << "Name: " << name << endl << "Multilabel: " << is_multilabel << endl << "Features: " << n_features << endl;
	os << "Train samples: " << n_train_samples << endl << "Test samples: " << n_test_samples << endl << "Validation samples: " << n_valid_samples << endl;
	os << "Response classes: " << n_response_classes << endl << "Time budget: " << time_budget << endl;
	return os.str();
}
