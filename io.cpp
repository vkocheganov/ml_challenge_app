#include "io.h"
#include "models.h"
#include <climits>

#include <fstream>
//#include <filesystem>
#include <algorithm>

#include <iostream>
using namespace std;

using namespace std;
const unsigned MISSED_VALUE_UINT = UINT_MAX;


//#include <windows.h>

// PVOID MapFile(LPCTSTR FileName, size_t* file_size)
// {
// 	DWORD dwAttr = GetFileAttributes(FileName);
// 	if (dwAttr == -1)
// 		return NULL;;

// 	HANDLE hFile = CreateFile(FileName, GENERIC_READ,
// 		FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, 0);

// 	if (hFile == INVALID_HANDLE_VALUE)
// 		return NULL;

// 	DWORD dwSizeHigh = 0;
// 	DWORD dwSizeLow = GetFileSize(hFile, &dwSizeHigh);
// 	*file_size = dwSizeLow;
// 	HANDLE hFileMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, dwSizeLow, NULL);
// 	CloseHandle(hFile);
// 	if (!hFileMap)
// 		return NULL;

// 	PVOID pAddr = MapViewOfFile(hFileMap, FILE_MAP_READ, 0, 0, dwSizeLow);

// 	if (NULL == pAddr)
// 		return NULL;
// 	CloseHandle(hFileMap);
// 	return pAddr;
// }

// struct FileInfo
// {
// 	bool is_valid;
// 	char* pfile;
// 	size_t file_size;
// 	FileInfo() : is_valid(false), pfile(nullptr), file_size(0) {}
// };


// FileInfo OpenFile(const fs::path file_name)
// {
// 	FileInfo file_info;
// 	if (!fs::exists(file_name))
// 	{
// 		printf("Error opening file: %s \n", file_name.string().c_str());
// 		return file_info;
// 	}

// 	file_info.pfile = (char*)MapFile(file_name.string().c_str(), &file_info.file_size);
// 	if (!file_info.pfile)
// 	{
// 		printf("Error mapping file: %s \n", file_name.string().c_str());
// 		return file_info;
// 	}
// 	file_info.is_valid = true;
// 	return file_info;
// }




// __forceinline int my_atoi(const char*& ptr, int& value, bool is_last_line)
// {
// 	int retcode = 1;
// 	/* skip whitespace */
// 	while (isspace((int)(unsigned char)*ptr))
// 	{
// 		if (*ptr == '\n')
// 		{
// 			if (is_last_line)
// 				return 0;
// 			retcode = 2;
// 		}
// 		++ptr;
// 	}

// 	int c = (int)(unsigned char)*ptr++;
// 	int sign = c;           /* save sign indication */
// 	if (c == '-' || c == '+')
// 		c = (int)(unsigned char)*ptr++;    /* skip sign */

// 	value = 0;
// 	if (!isdigit(c))
// 	{
// 		return 0;
// 	}
// 	while (isdigit(c))
// 	{
// 		value = 10 * value + (c - '0');     /* accumulate digit */
// 		c = (int)(unsigned char)*ptr++;    /* get next char */
// 	}

// 	if (sign == '-')
// 		value = -value;
// 	return retcode;
// }

// __forceinline float my_strtof(const char* &s, char decimal_separator = '.')
// {
// 	long double r;        /* result */
// 	int e;            /* exponent */
// 	long double d;        /* scale */
// 	int sign;         /* +- 1.0 */
// 	int esign;
// 	int i;
// 	int flags = 0;

// 	r = 0.0f;
// 	sign = 1;
// 	e = 0;
// 	esign = 1;

// 	while (isspace(*s))
// 		s++;

// 	if (*s == '+')
// 		s++;
// 	else if (*s == '-')
// 	{
// 		sign = -1;
// 		s++;
// 	}
// 	else if (*s == 'N')
// 	{
// 		s += 3;
// 		return MISSED_VALUE_FLOAT;
// 	}

// 	while ((*s >= '0') && (*s <= '9'))
// 	{
// 		flags |= 1;
// 		r *= 10.0L;
// 		r += *s - '0';
// 		s++;
// 	}

// 	if (*s == decimal_separator)
// 	{
// 		d = 0.1L;
// 		s++;
// 		while ((*s >= '0') && (*s <= '9'))
// 		{
// 			flags |= 2;
// 			r += d * (*s - '0');
// 			s++;
// 			d *= 0.1L;
// 		}
// 	}

// 	if (flags == 0)
// 	{
// 		return 0.0f;
// 	}

// 	if ((*s == 'e') || (*s == 'E'))
// 	{
// 		s++;
// 		if (*s == '+')
// 			s++;
// 		else if (*s == '-')
// 		{
// 			s++;
// 			esign = -1;
// 		}
// 		if ((*s < '0') || (*s > '9'))
// 		{
// 			return (float)r;
// 		}
// 		while ((*s >= '0') && (*s <= '9'))
// 		{
// 			e *= 10;
// 			e += *s - '0';
// 			s++;
// 		}
// 	}

// 	if (esign < 0)
// 		for (i = 1; i <= e; i++)
// 			r *= 0.1L;
// 	else
// 		for (i = 1; i <= e; i++)
// 			r *= 10.0L;

// 	return (float)(r * sign);
// }


#include <iostream>
//void FillPredictors(vector<float>& sample, const FileInfo& file_info, size_t n_variables, size_t n_samples, size_t sample_offset)
void FillPredictors(vector<float>& sample,  ifstream& file_info, size_t n_variables, size_t n_samples, size_t sample_offset)
{
	for (size_t i = 0; i < n_samples; ++i)
	{
		for (size_t j = 0; j < n_variables; ++j)
		{
                    float temp;
                    string tempStr;
                    file_info >> tempStr;
                    temp = stof(tempStr);
                    sample[n_variables * (i + sample_offset) + j] = isnan(temp) ? FLT_MAX : temp;
		}
	}
//        std::cout<<"sample[5*n_variables + 3] = "<<sample[5*n_variables + 3]<<std::endl;
//        std::cout<<"sample[5*n_variables + 2] = "<<sample[5*n_variables + 2]<<std::endl;
}

#include <sstream>
#include <string>
//void FillPredictorsSparseBinary(vector<float>& sample, const FileInfo& file_info, size_t n_variables, size_t n_samples, size_t sample_offset)
void FillPredictorsSparseBinary(vector<float>& sample, ifstream& file_info, size_t n_variables, size_t n_samples, size_t sample_offset)
{
	size_t row = 0;
	int feature;
        
        string line;
        while (getline(file_info,line))
        {
            std::istringstream iss(line);
            while (iss >> feature)
            {
                --feature;
                sample[n_variables * (row + sample_offset) + feature] = 1.f;
            }
            row++;
        }
}


//void FillPredictorsSparse(vector<float>& sample, const FileInfo& file_info, size_t n_variables, size_t n_samples, size_t sample_offset)
void FillPredictorsSparse(vector<float>& sample, ifstream& file_info, size_t n_variables, size_t n_samples, size_t sample_offset)
{
	size_t row = 0;
	int feature, value;
        
        string line;
        while (getline(file_info,line))
        {
            std::istringstream iss(line);
            string temp1,temp2;
            int feature = 0;
            while (getline(iss,temp1,':')>>temp2)
            {
                feature = stoi(temp1) - 1;
                float temp = stof(temp2);
                sample[n_variables * (row + sample_offset) + feature] = isnan(temp) ? FLT_MAX : temp;
            }
            row++;
        }
}

//void FillResponses(vector<vector<float>>& responses, size_t n_responses, size_t n_samples, size_t sample_offset, bool is_multilabel, const FileInfo& file_info, vector<int>& labelsMap)
void FillResponses(vector<vector<float>>& responses, size_t n_responses, size_t n_samples, size_t sample_offset, bool is_multilabel, ifstream& file_info, vector<int>& labelsMap)
{
	labelsMap.clear();
	bool is_binary = (n_responses == 2);
	if (n_responses == 0 || n_responses == 2)
	{
		// handle binary classification and regression
		n_responses = 1;
	}

	std::vector<float> _responses(n_responses, 0);
	for (size_t i = 0; i < n_samples; ++i)
	{
		for (size_t j = 0; j < n_responses; ++j)
                    file_info>>_responses[j];

		if (is_multilabel)
		{//adult
			// multilabel classification
			for (size_t j = 0; j < n_responses; ++j)
				responses[j][sample_offset + i] = _responses[j];
		}
		else if (n_responses > 2)
		{// digits, newsgroups
			// multiclass classification
			size_t class_idx = max_element(_responses.begin(), _responses.end()) - _responses.begin();
			responses[0][sample_offset + i] = (float)class_idx;
			if (labelsMap.size() < n_responses)
			{
				auto it = find(labelsMap.begin(), labelsMap.end(), class_idx);
				if (it == labelsMap.end())
				{
					labelsMap.push_back(class_idx);
				}
				else
				{
					if (class_idx != *it)
                                            exit(1);
				}
			}
		}
		else
		{//cadata, dorothea
			// binary classification, multiclass classification and regression
//			sample._responses[0][sample_offset + i].fl = responses[0];
			responses[0][sample_offset + i] = _responses[0];
		}
	}
	//sort(labelsMap.begin(), labelsMap.end());
	if (is_binary)
	{
		labelsMap.push_back(responses[0][0] != 0);
		labelsMap.push_back(responses[0][0] == 0);
	}
}



//void DumpResults(const vector<float>&predictions, const vector<float>& class_probs, const string& _output_dir, const SampleInfo& sample_info)
void DumpResults(const vector<float>&predictions, const vector<float>& class_probs, const string& output_dir, const SampleInfo& sample_info)
{
	ofstream valid_file(output_dir +"/"+sample_info.name + "_valid_001.predict");
	ofstream test_file(output_dir +"/"+sample_info.name + "_test_001.predict");
	size_t n_samples = sample_info.n_valid_samples + sample_info.n_test_samples;
        
	if (sample_info.is_multilabel)
	{//adult
		// multilabel classification: for each label, write class 1 probability
		for (size_t i = 0; i < sample_info.n_valid_samples; ++i)
		{
			for (size_t j = 0; j < sample_info.n_response_classes; ++j)
			{
				float resp = class_probs[2 * j * n_samples + 2 * i + 1];
				valid_file << resp << " ";
			}
			valid_file << endl;
		}

		for (size_t i = 0; i < sample_info.n_test_samples; ++i)
		{
			for (size_t j = 0; j < sample_info.n_response_classes; ++j)
			{
				float resp = class_probs[2 * j * n_samples + 2 * (i + sample_info.n_valid_samples) + 1];
				test_file << resp << " ";
			}
			test_file << endl;
		}
	}
	else if (sample_info.n_response_classes > 2)
	{//digits, newsgroups
		// multiclass classification: write probability of each class
		for (size_t i = 0; i < sample_info.n_valid_samples; ++i)
		{
			for (size_t j = 0; j < sample_info.n_response_classes; ++j)
			{
				float resp = class_probs[sample_info.n_response_classes * i + j];
				valid_file << resp << " ";
			}
			valid_file << endl;
		}

		for (size_t i = 0; i < sample_info.n_test_samples; ++i)
		{
			for (size_t j = 0; j < sample_info.n_response_classes; ++j)
			{
				float resp = class_probs[sample_info.n_response_classes * (i + sample_info.n_valid_samples) + j];
				test_file << resp << " ";
			}
			test_file << endl;
		}
	}
	else if (sample_info.n_response_classes == 2)
	{//dorothea
		// binary classification: write class 1 probability
		for (size_t i = 0; i < sample_info.n_valid_samples; ++i)
			valid_file << class_probs[2 * i + 1] << endl;

		for (size_t i = 0; i < sample_info.n_test_samples; ++i)
			test_file << class_probs[2 * (i + sample_info.n_valid_samples) + 1] << endl;
	}
	else
	{//cadata
		// regression : write predicted value
		for (size_t i = 0; i < sample_info.n_valid_samples; ++i)
			valid_file << predictions[i] << endl;

		for (size_t i = 0; i < sample_info.n_test_samples; ++i)
			test_file << predictions[i + sample_info.n_valid_samples] << endl;
	}
}



//void CreateSample(const std::string& _input_dir, const SampleInfo& sample_info, vector<float>& sample, vector<vector<float>>& responses, vector<int>& labelsMap)
void CreateSample(const std::string& input_dir, const SampleInfo& sample_info, vector<float>& sample, vector<vector<float>>& responses, vector<int>& labelsMap)
{
	string basename(sample_info.name);
	ifstream train_file_info(input_dir + "/" + basename + "/" + basename+ "_train.data");
	ifstream valid_file_info(input_dir + "/" + basename + "/" + basename+ "_valid.data");
	ifstream test_file_info(input_dir + "/" + basename +"/" + basename+  "_test.data");
	ifstream y_file_info(input_dir + "/" + basename +"/" + basename+  "_train.solution");
        if (!train_file_info.is_open())
            cout<<"train_file failed to open"<<endl;
        if (!valid_file_info.is_open())
            cout<<"valid_file failed to open"<<endl;
        if (!test_file_info.is_open())
            cout<<"test_file failed to open"<<endl;
        if (!y_file_info.is_open())
            cout<<"y_file failed to open"<<endl;

	size_t n_responses = sample_info.is_multilabel ? sample_info.n_response_classes : 1;
	size_t n_samples = sample_info.n_train_samples + sample_info.n_test_samples + sample_info.n_valid_samples;
	responses.resize(n_responses == 0 ? 1 : n_responses);
	for (auto& r : responses)
	{
		r.resize(sample_info.n_train_samples);
	}
	//sample.AllocateValuesAndTypes(sample_info.n_features, n_samples, n_responses);
	sample.resize(n_samples*sample_info.n_features);
	if (sample_info.is_sparse)
	{//dorothea, newsgroups
		if (sample_info.features_type == FeaturesType::Binary)
		{//dorothea
			// sparse matrix with indices of ones
			fill(sample.begin(), sample.end(), 0);
			FillPredictorsSparseBinary(sample, train_file_info, sample_info.n_features, sample_info.n_train_samples, 0);
			FillPredictorsSparseBinary(sample, valid_file_info, sample_info.n_features, sample_info.n_valid_samples, sample_info.n_train_samples);
			FillPredictorsSparseBinary(sample, test_file_info, sample_info.n_features, sample_info.n_test_samples, sample_info.n_train_samples + sample_info.n_valid_samples);
		}
		else
		{//newsgroups
			// sparse matrix with indices of ones
			fill(sample.begin(), sample.end(), 0);
			FillPredictorsSparse(sample, train_file_info, sample_info.n_features, sample_info.n_train_samples, 0);
			FillPredictorsSparse(sample, valid_file_info, sample_info.n_features, sample_info.n_valid_samples, sample_info.n_train_samples);
			FillPredictorsSparse(sample, test_file_info, sample_info.n_features, sample_info.n_test_samples, sample_info.n_train_samples + sample_info.n_valid_samples);
		}
	}
	else
	{//adult, cadata, digits
		FillPredictors(sample, train_file_info, sample_info.n_features, sample_info.n_train_samples, 0);
		FillPredictors(sample, valid_file_info, sample_info.n_features, sample_info.n_valid_samples, sample_info.n_train_samples);
		FillPredictors(sample, test_file_info, sample_info.n_features, sample_info.n_test_samples, sample_info.n_train_samples + sample_info.n_valid_samples);
	}

	FillResponses(responses, sample_info.n_response_classes, sample_info.n_train_samples, 0, sample_info.is_multilabel, y_file_info, labelsMap);

	// specify all predictors to be binary categorical and response to be multilevel categorical
	//FillTypeMask(input_dir, sample_info, sample);

}



