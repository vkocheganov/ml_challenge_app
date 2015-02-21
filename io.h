#pragma once
#include "properties.h"
#include <vector>

using namespace std;

void CreateSample(const std::string& input_dir, const SampleInfo& sample_info, vector<float>&, vector<vector<float>>&, vector<int>&);
void DumpResults(const vector<float>&predictions, const vector<float>& class_probs, const string& _output_dir, const SampleInfo& sample_info);

