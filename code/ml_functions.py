import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from functions import count_mode, bayes_probs, bayes_binary
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F



def crowd_size(data, target = 'mode', size = 10):
	
	all_cols = data.columns
	
	resp_list = [x for x in all_cols if 'resp_cat' in x]
	resp_var = [x for x in all_cols if 'resp_veracity' in x]
	resp_ideo = [x for x in all_cols if 'ideology_resp' in x]
	
	resp_list = resp_list[:size]
	resp_var = resp_var[:size]
	resp_ideo = resp_ideo[:size]
	
	cols = []
	
	cols.extend(resp_list)
	cols.extend(resp_var)
	cols.extend(resp_ideo)
	cols.append(target)
	
	return data.loc[:,cols]


def feature_transform(data, train = None):
	
	
	all_cols = data.columns
	
	
	resp_list = [x for x in all_cols if 'resp_cat' in x]
	resp_var = [x for x in all_cols if 'resp_veracity' in x]
	resp_ideo = [x for x in all_cols if 'ideology_resp' in x]
	
	#data size
	data_size = len(resp_list)
	
	#getting features
	counters = data[resp_list].apply(Counter, axis = 1)


	#new features
	crowd_t = counters.map(lambda x:x['t'])
	crowd_f = counters.map(lambda x:x['f'])
	crowd_c = counters.map(lambda x:x['c'])
	
	t_f = crowd_t - crowd_f
	t_f_d = crowd_t/(crowd_f +1)
	c_dif = crowd_c - crowd_t - crowd_f
	t_dif = crowd_t - crowd_c - crowd_f
	f_dif = crowd_f - crowd_t - crowd_c
	c_per = crowd_c/(crowd_f + crowd_c + crowd_t)
	t_per = crowd_t/(crowd_f + crowd_c + crowd_t)
	f_per = crowd_f/(crowd_f + crowd_c + crowd_t)
	f_rat = (crowd_f + crowd_c) / (crowd_t + 1)
	t_rat = (crowd_t + crowd_c) / (crowd_f + 1)


	modes = counters.apply(count_mode)
	
	means = data[resp_var].mean(axis = 1)
	
	median = data[resp_var].median(axis = 1)
	
	full_range = data[resp_var].quantile(1, axis = 1) - data[resp_var].quantile(0, axis = 1)
	
	IQR_range = data[resp_var].quantile(.75, axis = 1) - data[resp_var].quantile(.25, axis = 1)
	
	variance = data[resp_var].var(axis = 1)

	
	
	#bayes stuff
	if train is None:
		
		
		bayes_dic = bayes_probs(data, "fcmodefalse", num_resp = data_size)

	else:
		
		bayes_dic = bayes_probs(train, "fcmodefalse", num_resp = data_size)
		
		
	listed_data = data[resp_list].iloc[:,:].apply(list, axis = 1)

	bayes_preds = listed_data.apply(bayes_binary, bayes_dic_input = bayes_dic, fc_type = "fcmodefalse", num_resp = data_size,return_prob = True)
	

	return_data = data.copy()
	return_data['crowd_mode'] = modes
	return_data['crowd_means'] = means
	return_data['crowd_median'] = median
	return_data['crowd_full_range'] = full_range
	return_data['crowd_IQR_range'] = IQR_range
	return_data['crowd_variance'] = variance
	return_data['crowd_bayes'] = bayes_preds

	return_data['new_crowd_t'] = crowd_t
	return_data['new_crowd_f'] = crowd_f
	return_data['new_crowd_c'] = crowd_c
	return_data['new_t_f'] = t_f
	return_data['new_t_f_d'] = t_f_d
	return_data['new_c_dif'] = c_dif
	return_data['new_t_dif'] = t_dif
	return_data['new_f_dif'] = f_dif
	return_data['new_c_per'] = c_per
	return_data['new_t_per'] = t_per
	return_data['new_f_per'] = f_per
	return_data['new_f_rat'] = f_rat
	return_data['new_t_rat'] = t_rat
	
	
	return return_data


def convert_cat(data):
	
	all_cols = data.columns
	
	resp_list = [x for x in all_cols if 'resp_cat' in x]
	
	resp_list.append('crowd_mode')
	
	dummies = pd.get_dummies(data[resp_list], drop_first = "true")
	
	return dummies       


def feature_creation(data, crowd_num = 10, target = "mode", train = None):
	
	
	#deleting columns
	keepers = []
	
	all_cols = data.columns
	
	resp_list = [x for x in all_cols if 'resp_cat' in x]
	resp_var = [x for x in all_cols if 'resp_veracity' in x]
	resp_ideo = [x for x in all_cols if 'ideology_resp' in x]
	
	keepers.extend(resp_list)
	keepers.extend(resp_var)
	keepers.extend(resp_ideo)
	
	keepers.append(target)
	
	keepers = set(keepers)
	
	col_set = set(all_cols)

	those_to_lose = col_set - keepers 
	
	data_alt = data.copy()

	for col in those_to_lose:
		
		del data_alt[col]
		  
	#adding data
	new_data = crowd_size(data_alt, size = crowd_num)
	new_data = feature_transform(new_data,train = train)
	
	#getting dummies and adding them
	dummies = convert_cat(new_data)
	new_data = new_data.join(dummies)
	
	for col_x in resp_list[:crowd_num]:
		
		del new_data[col_x]
		
		
	del new_data[target]
	del new_data['crowd_mode']
	
	return new_data


def conf_eval(preds, truth):
	
	if True in np.unique(truth):
		
		condition_pos = True
		condition_neg = False
		
	else:
		
		condition_pos = 1
		condition_neg = 0
		
	con_pos_count = np.sum(truth == condition_pos)
	con_neg_count = np.sum(truth == condition_neg)
		
	accuracy = np.sum((preds == truth))/truth.size
	
	True_pos = np.sum(preds[truth == condition_pos] == truth[truth ==condition_pos])/con_pos_count
	False_pos = np.sum(preds[truth == condition_pos] != truth[truth ==condition_pos])/con_pos_count
	
	True_neg = np.sum(preds[truth == condition_neg] == truth[truth == condition_neg])/con_neg_count
	False_neg = np.sum(preds[truth == condition_neg] != truth[truth == condition_neg])/con_neg_count
	
	
	return accuracy, True_pos, False_pos, True_neg,False_neg
	
	

class data_prep(Dataset):
	"""
	Class that represents a train/validation/test dataset that's readable for PyTorch
	Note that this class inherits torch.utils.data.Dataset
	"""
	
	def __init__(self, data_list, target_list):
		"""
		@param data_list: list of newsgroup tokens 
		@param target_list: list of newsgroup targets 

		"""
		self.data_list = data_list.values
		self.target_list = target_list
		assert (len(self.data_list) == len(self.target_list))

	def __len__(self):
		return len(self.data_list)
		
	def __getitem__(self, key):
		"""
		Triggered when you call dataset[i]
		"""
		
		data = self.data_list[key]
		label = self.target_list[key]
		return [data, label]
	
	
# Function for testing the model
def test_model(loader, model):
	"""
	Help function that tests the model's performance on a dataset
	@param: loader - data loader for the dataset to test against
	"""
	correct = 0
	total = 0
	model.eval()
	for data, labels in loader:
		data_batch, label_batch = data, labels
		outputs = torch.sigmoid(model(data_batch.float()))
		predicted = np.where(outputs > .5,1,0)
		np_labels = labels.numpy().reshape(-1,1)
		
		total += labels.size(0)
		correct += np.sum(predicted == np_labels)
	return (100 * correct / total)
	
	
	