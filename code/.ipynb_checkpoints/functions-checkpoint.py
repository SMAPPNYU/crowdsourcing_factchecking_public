"""
This notebook is where the functions that are used for heuristics are created and maintained
"""

import numpy as np 
import pandas as pd 
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import multinomial



def count_mode(count_dic):
	#this function counts the mode from a dictionary provided to it

	count_dic.pop(np.nan, None)

	max_num = count_dic.most_common(1)[0][1]

	all_max = []

	for keys_count, nums_count in count_dic.items():

		if nums_count == max_num:

			all_max.append(keys_count)

	if "t" in all_max:

		return 't'

	elif len(all_max) == 1:

		return all_max[0]

	else:

		return 'f' 


### this function returns mode from a counter
### it ignores CND
def count_func(count_dic):

	count_dic.pop(np.nan, None)

	if count_dic['t'] >= count_dic['f']:

		return 't'

	elif count_dic['t'] < count_dic['f']:

		return 'f'

### this function turens mode from a counter
### it treats CND as FM
def all_fm(count_dic):

	count_dic.pop(np.nan, None)

	if count_dic['f'] + count_dic['c'] > count_dic['t']:

		return 'f'

	elif count_dic['f'] + count_dic['c'] <= count_dic['t']:
	
		return 't'

def unan(count_dic):

	count_dic.pop(np.nan, None)

	if len(count_dic) == 1:

		return list(count_dic.keys())[0]
    
	else: 

		return 'c'



### this function extracts the crowd mode based on three criteria
def crowd_mode(df, cnd = "include", do_count = False, bayes = None):
	"""
	For categorical mode of crowd
	cnd
	include: Include CND
	discard: Discard CND
	fm: treat at FM


	Bayes shoukd be FC_type and number of respondents as a list 
	"""
	if do_count == False:

		counters = df.apply(Counter, axis = 1)

	else:

		counters = df


	if bayes is not None:


		bayes_dic = bayes_probs(df, bayes[0])

		if len(bayes) ==3:

			short_data = bayes[2].iloc[:,:bayes[1]]

		else:

			short_data = df.iloc[:,:bayes[1]]

		listed_data = short_data.iloc[:,:].apply(list, axis = 1)

		preds = listed_data.apply(bayes_binary, bayes_dic_input = bayes_dic, fc_type = bayes[0], num_resp = bayes[1])

		return preds


	if cnd == "include":

		modes = counters.apply(count_mode)

		return modes #this is a series

	elif cnd == "discard":

		modes = counters.apply(count_func) #this is a series

		return modes

	elif cnd == "fm":

		modes = counters.apply(all_fm)

		return modes

	elif cnd == "unan":

		modes = counters.apply(unan)

		return modes #this is a series



def eval_comp(answers,df, fc_type,eval_col):
	"""
	Takes answer from crowd
	and df with FC classifications
	and the column of the classification
	"""
	mode_dic = {'T':"t", 'FM':"f", 'No Mode!':'c', 'CND':'c'} # no mode is treated as CND here


	false_set = set(["no_true", "all_false_set", "any_false_set"])
	true_set = set(["no_false", "all_true_set", "any_true_set"])

	assert answers.shape[0] == df.shape[0]

	answers.reset_index(inplace = True, drop = True)
	df.reset_index(inplace = True, drop = True)

	# pure mode
	if eval_col == "mode" and "false" in fc_type:

		mode_series = df[eval_col].copy()
		mode_series.reset_index(inplace = True, drop = True)

		#correct for false
		true_pos = np.where(answers.eq("f") & mode_series.eq("FM"),True, False)

		#correct for not false 
		true_neg = np.where(answers.ne("f") & mode_series.ne('FM'),True, False)

		#num_correct = np.sum(new_col1) + np.sum(new_col2)

		false_pos = np.where(answers.eq("f") & mode_series.ne("FM"),True, False)

		false_neg = np.where(answers.ne("f") & mode_series.eq('FM'),True, False)

		baseline = max([mode_series.eq("FM").sum(), mode_series.ne("FM").sum()])

		total = df.shape[0]

		return np.sum(true_pos), np.sum(true_neg), np.sum(false_pos), np.sum(false_neg), total, baseline

	elif eval_col == "mode" and "true" in fc_type:

		mode_series = df[eval_col].copy()
		mode_series.reset_index(inplace = True, drop = True)

		#correct for true
		true_pos = np.where(answers.eq("t") & mode_series.eq("T"),True, False)

		#correct for not true 
		true_neg = np.where(answers.ne("t") & mode_series.ne('T'),True, False)

		#num_correct = np.sum(new_col1) + np.sum(new_col2)

		false_pos = np.where(answers.eq("t") & mode_series.ne("T"),True, False)

		false_neg = np.where(answers.ne("t") & mode_series.eq('T'),True, False)

		baseline = max([mode_series.eq("T").sum(), mode_series.ne("T").sum()])

		total = df.shape[0]

		return np.sum(true_pos), np.sum(true_neg), np.sum(false_pos), np.sum(false_neg), total, baseline
   
	#robust mode
	elif eval_col == "in_robust_mode" and "false" in fc_type:

		robust_mode = np.where(df["mode"].eq("FM") & df["in_robust_mode"].eq(True), "f","t")

		robust_mode = pd.Series(robust_mode)

		true_pos = np.where(answers.eq("f") & robust_mode.eq("f"),True, False)

		true_neg = np.where(answers.ne("f") & robust_mode.ne("f"),True, False)

		#num_correct = np.sum(new_col1) + np.sum(new_col2)

		false_pos = np.where(answers.eq("f") & robust_mode.ne("f"),True, False)

		false_neg = np.where(answers.ne("f") & robust_mode.eq("f"),True, False)

		baseline = max([robust_mode.eq("f").sum(), robust_mode.ne("f").sum()])

		total = df.shape[0]

		return np.sum(true_pos), np.sum(true_neg), np.sum(false_pos), np.sum(false_neg), total, baseline

	elif eval_col == "in_robust_mode" and "true" in fc_type:

		robust_mode = np.where(df["mode"].eq("T") & df["in_robust_mode"].eq(True), "t","f")

		robust_mode = pd.Series(robust_mode)

		#correct for false
		true_pos = np.where(answers.eq("t") & robust_mode.eq("t"),True, False)

		#correct for not false 
		true_neg = np.where(answers.ne("t") & robust_mode.ne("t"),True, False)

		#num_correct = np.sum(new_col1) + np.sum(new_col2)

		false_pos = np.where(answers.eq("t") & robust_mode.ne("t"),True, False)

		false_neg = np.where(answers.ne("t") & robust_mode.eq("t"),True, False)

		baseline = max([robust_mode.eq("t").sum(), robust_mode.ne("t").sum()])

		total = df.shape[0]

		return np.sum(true_pos), np.sum(true_neg), np.sum(false_pos), np.sum(false_neg), total, baseline

	else:

		#eval col is one of the false sets
		if eval_col in false_set:

			eval_series = df[eval_col].copy()

			eval_series.reset_index(inplace = True, drop = True)

			true_pos = np.where(answers.eq("f") & eval_series.eq(True),True, False)

			true_neg = np.where(answers.ne("f") & eval_series.eq(False),True, False)

			#final = np.sum(new_col1) + np.sum(new_col2)

			false_pos = np.where(answers.eq("f") & eval_series.ne(True),True, False)

			false_neg = np.where(answers.ne("f") & eval_series.ne(False),True, False)

			total = df.shape[0]

			baseline = max([eval_series.eq(True).sum(), eval_series.eq(False).sum()])

			return np.sum(true_pos), np.sum(true_neg), np.sum(false_pos), np.sum(false_neg), total, baseline

		elif eval_col in true_set: # if it is one of the true sets

			eval_series = df[eval_col].copy()

			eval_series.reset_index(inplace = True, drop = True)
			
			true_pos = np.where(answers.eq("t") & eval_series.eq(True),True, False)

			true_neg = np.where(answers.ne("t") & eval_series.eq(False),True, False)

			#final = np.sum(new_col1) + np.sum(new_col2)

			false_pos = np.where(answers.eq("t") & eval_series.ne(True),True, False)

			false_neg = np.where(answers.ne("t") & eval_series.ne(False),True, False)

			baseline = max([eval_series.eq(True).sum(), eval_series.eq(False).sum()])

			total = df.shape[0]

			return np.sum(true_pos), np.sum(true_neg), np.sum(false_pos), np.sum(false_neg), total, baseline


def full_evaluation(df,full_df, fc_type,crowd_type, do_count1 = False, bayes = None, num_resp = 1, test_data = None):

	#necessary dictionaries
	cnd_dic = {'origmode':"include", 'nocnd':"discard", 'modefm':'fm', 'unan':'unan'}

	types_to_cols= {"allfalse":"all_false_set", 
	"anyfalse":"any_false_set", 
	"notrue":"no_true", 
	"alltrue":"all_true_set", 
	"anytrue":"any_true_set", 
	"nofalse":"no_false",
	"fcmodefalse":"mode",
	"rmfalse":"in_robust_mode",
	"fcmodetrue":"mode",
	"rmtrue":"in_robust_mode"}


	eval_col = types_to_cols[fc_type]


	### DF subsetting 
	if bayes is not None:

		if test_data is None:

			crowd_answers = crowd_mode(full_df, do_count = True, bayes = [bayes,num_resp])

			true_pos, true_neg, false_pos, false_neg, total, baseline = eval_comp(crowd_answers,full_df, fc_type, eval_col)


		else:

			crowd_answers = crowd_mode(full_df, do_count = True, bayes = [bayes,num_resp, test_data])

			true_pos, true_neg, false_pos, false_neg, total, baseline = eval_comp(crowd_answers,test_data, fc_type, eval_col)


	else:

		crowd_answers = crowd_mode(df, cnd = cnd_dic[crowd_type], do_count = do_count1)

		true_pos, true_neg, false_pos, false_neg, total, baseline = eval_comp(crowd_answers,full_df, fc_type, eval_col)

	return true_pos, true_neg, false_pos, false_neg, total, baseline




def array_return(input_array, length = 2):

	keep_list = []
	l_count = 0
	c_count = 0
	m_count = 0

	for this_person in input_array:

		if this_person < 0 and l_count < length:

			keep_list.append(True)

			l_count += 1

		elif this_person > 0 and c_count < length:

			c_count += 1

			keep_list.append(True)

		elif this_person == 0 and m_count < length:

			keep_list.append(True)

			m_count += 1

		else: 

			keep_list.append(False)

	return keep_list


#this function calculates probabiities for bayes

def bayes_probs(bayes_df, fc_type, num_resp = 30):

	bayes_dic = {} #save it all here

	types_to_cols= {"allfalse":"all_false_set", 
	"anyfalse":"any_false_set", 
	"notrue":"no_true", 
	"alltrue":"all_true_set", 
	"anytrue":"any_true_set", 
	"nofalse":"no_false",
	"fcmodefalse":"mode",
	"rmfalse":"in_robust_mode",
	"fcmodetrue":"mode",
	"rmtrue":"in_robust_mode"}

	bayes_df.reset_index(inplace = True, drop = True)

	#print(bayes_df)

	total_lines = bayes_df.shape[0]

	false_set = set(["allfalse", "anyfalse", "notrue", "fcmodefalse", "rmfalse"])
	true_set = set(["alltrue", "anytrue", "nofalse", "fcmodetrue", "rmtrue"])

	eval_col = types_to_cols[fc_type]

	#first calculate prior
	if eval_col == "in_robust_mode" and "false" in fc_type:

		robust_mode = np.where(bayes_df["mode"].eq("FM") & bayes_df["in_robust_mode"].eq(True), True, False)

		bayes_dic['prior'] =  np.sum(robust_mode)/total_lines

	elif eval_col == "in_robust_mode" and "true" in fc_type:

		robust_mode = np.where(bayes_df["mode"].eq("T") & bayes_df["in_robust_mode"].eq(True), True, False)

		bayes_dic['prior'] = np.sum(robust_mode)/total_lines

	elif eval_col == "mode" and "false" in fc_type:

		bayes_dic['prior'] = np.sum(bayes_df["mode"].eq("FM"))/total_lines

	elif eval_col == "mode" and "true" in fc_type:

		bayes_dic['prior'] = np.sum(bayes_df["mode"].eq("T"))/total_lines

	else:

		bayes_dic['prior'] = np.sum(bayes_df[eval_col])/total_lines


	#calculating conditionals
	#first subsetting to DFs where FC called it either true or false
	if eval_col == "in_robust_mode" and "false" in fc_type:

		robust_mode = np.where(bayes_df["mode"].eq("FM") & bayes_df["in_robust_mode"].eq(True), True, False)

		bayes_df_pos = bayes_df.loc[robust_mode,:].copy()
		bayes_df_neg = bayes_df.loc[~robust_mode,:].copy()

	elif eval_col == "in_robust_mode" and "true" in fc_type:

		robust_mode = np.where(bayes_df["mode"].eq("T") & bayes_df["in_robust_mode"].eq(True), True, False)

		bayes_df_pos = bayes_df.loc[robust_mode,:].copy()
		bayes_df_neg = bayes_df.loc[~robust_mode,:].copy()

	elif eval_col == "mode" and "false" in fc_type:
 
		bayes_df_pos = bayes_df.loc[bayes_df[eval_col].eq("FM"),:].copy()
		bayes_df_neg = bayes_df.loc[bayes_df[eval_col].ne("FM"),:].copy()

	elif eval_col == "mode" and "true" in fc_type:

		bayes_df_pos = bayes_df.loc[bayes_df[eval_col].eq("T"),:].copy()
		bayes_df_neg = bayes_df.loc[bayes_df[eval_col].ne("T"),:].copy()

	else:

		bayes_df_pos = bayes_df.loc[bayes_df[eval_col] == True,:].copy()
		bayes_df_neg = bayes_df.loc[bayes_df[eval_col] == False,:].copy()



	total_pos = bayes_df_pos.iloc[:,:num_resp].size
	total_neg = bayes_df_neg.iloc[:,:num_resp].size

	bayes_df_pos_sub = bayes_df_pos.iloc[:,:num_resp]
	bayes_df_neg_sub = bayes_df_neg.iloc[:,:num_resp]


	if fc_type in false_set:
	
		num_correct_pos = (bayes_df_pos_sub == "f").sum().sum()
		num_cnd_pos = (bayes_df_pos_sub == "c").sum().sum()
		num_incorrect_pos = (bayes_df_pos_sub == "t").sum().sum()

		assert num_correct_pos + num_cnd_pos +  num_incorrect_pos == total_pos

		num_correct_neg = (bayes_df_neg_sub == "t").sum().sum()
		num_cnd_neg = (bayes_df_neg_sub == "c").sum().sum()
		num_incorrect_neg = (bayes_df_neg_sub == "f").sum().sum()

		assert num_correct_neg + num_cnd_neg+  num_incorrect_neg == total_neg


	else:

		num_correct_pos = (bayes_df_pos_sub == "t").sum().sum()
		num_cnd_pos = (bayes_df_pos_sub == "c").sum().sum()
		num_incorrect_pos = (bayes_df_pos_sub == "f").sum().sum()

		assert num_correct_pos + num_cnd_pos +  num_incorrect_pos == total_pos


		num_correct_neg = (bayes_df_neg_sub == "f").sum().sum()
		num_cnd_neg = (bayes_df_neg_sub == "c").sum().sum()
		num_incorrect_neg = (bayes_df_neg_sub == "t").sum().sum()

		assert num_correct_neg + num_cnd_neg+  num_incorrect_neg == total_neg


	bayes_dic['correct|positive'] = num_correct_pos/total_pos
	bayes_dic['cnd|positive'] = num_cnd_pos/total_pos
	bayes_dic['incorrect|positive'] = num_incorrect_pos/total_pos

	bayes_dic['correct|negative'] = num_correct_neg/total_neg
	bayes_dic['cnd|negative'] = num_cnd_neg/total_neg
	bayes_dic['incorrect|negative'] = num_incorrect_neg/total_neg


	return bayes_dic


# This function performes bayes probability given a list
#simple - only have true or false

def bayes_binary(response_list, bayes_dic_input, fc_type, num_resp = 1, return_prob = False):

	false_set = set(["allfalse", "anyfalse", "notrue", "fcmodefalse", "rmfalse"])
	true_set = set(["alltrue", "anytrue", "nofalse", "fcmodetrue", "rmtrue"])

	if fc_type in true_set:

		type_set = True

	else:

		type_set = True


	#default is probability it is false
	prior = bayes_dic_input['prior']
	
	numerator = prior

	relavant_responses = response_list[:num_resp]

	if num_resp > len(relavant_responses):


		print("not as many respondents as assumed")
		
	#calculating denominator
	response_count = Counter(relavant_responses)
	#print(response_count)
	#calculating true multinomial
	prob_list = [bayes_dic_input['correct|positive'],bayes_dic_input['incorrect|positive'],bayes_dic_input['cnd|positive']]

	#calculating false multinomial
	prob_list_false = [bayes_dic_input['incorrect|negative'],bayes_dic_input['correct|negative'],bayes_dic_input['cnd|negative']]

	#  Both _true because both are the "True" condition
	if not true_set:

		multi = multinomial(len(relavant_responses), prob_list)
		prob_multi_true = multi.pmf([response_count['f'], response_count['t'], response_count['c']])

		false_multi = multinomial(len(relavant_responses), prob_list_false)
		prob_multi_false= false_multi.pmf([response_count['f'], response_count['t'], response_count['c']])

	else:

		multi = multinomial(len(relavant_responses), prob_list)
		prob_multi_true = multi.pmf([response_count['t'], response_count['f'], response_count['c']])
		#print([response_count['t'], response_count['f'], response_count['c']])

		false_multi = multinomial(len(relavant_responses), prob_list_false)
		prob_multi_false= false_multi.pmf([response_count['t'], response_count['f'], response_count['c']])
		#print(prob_multi_false)

	denominator = (1 - prior)*prob_multi_false + prior*prob_multi_true
	
	numerator = numerator*prob_multi_true

	if return_prob:

		return numerator / denominator 

	else:

		prob = numerator / denominator 

		if type_set:

			if prob >= .5:

				return 't'

			else:

				return 'f'

		else:

			if prob >= .5:

				return 'f'

			else:

				return 't'









def quick_graph(double_dic, double_key, title = None, 
				column = "percentage", partisan = False, baseline = False,
				evens = False, standard_y= False):

	mode_label_dic = {"origmode":"Mode", "nocnd":"No CND", "modefm": "CND as FM","unanimous": "Unanimous","bayes":"Bayes"}

	fc_label_dic = {'fcmodefalse':"Mode False",
	"rmfalse": "Robust Mode False",
	"allfalse": "All False",
	"anyfalse": "Any False",
	"notrue": "No True",
	"alltrue":"All True",
	'fcmodetrue':" Mode True",
	"rmtrue":"Robust Mode True",
	"anytrue": "Any True",
	"nofalse": "No False"}

	double_list = double_dic[double_key] 

	df= pd.DataFrame(double_list, columns = ["true_pos", "true_neg", "false_pos", "false_neg", "total", "baseline"])
	df['percentage'] = (df.true_pos+ df.true_neg)/df.total

	if evens:

		evens = list(range(0,24,2))

		evens_plus = [x+1 for x in evens]

		x_axis = evens_plus

		df = df.loc[evens,:]

	elif partisan == False:
	
		x_axis = list(range(1,len(double_list) + 1))

	else:

		x_axis = list(range(3,3*(len(double_list) + 1),3))


	label = double_key.split("_")

	new_label = fc_label_dic[label[0]] + "  -  " + mode_label_dic[label[1]]

	plt.ylabel("Percent Correct")
	plt.xlabel("Crowd Size")



	if standard_y != False:

		plt.ylim(standard_y[0], standard_y[1])

	if title is not None:

		plt.title(title)

	#df.percentage.plot(label = new_label)

	if column ==  "percentage":
		plt.plot(x_axis,df[column], label = new_label, linewidth=4)
		plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 3))
		#plt.legend(loc="upper left", bbox_to_anchor=(1,1))

		if baseline == True:

			df['baseline_percent'] = df.baseline/df.total

			plt.plot(x_axis,df['baseline_percent'],'--', label = new_label + " Baseline", color = "red")
		#plt.legend(loc="upper left", bbox_to_anchor=(1,1))


	else:
		new_label = new_label  + "  -  " + column
		plt.plot(x_axis,df[column]/df.total, label = new_label, linewidth=4)
		#plt.legend(loc="upper left", bbox_to_anchor=(1,1))

	if partisan:
		plt.scatter(x_axis,df[column],marker='o')

		plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 3))

		if baseline == True:

			df['baseline_percent'] = df.baseline/df.total

			plt.plot(x_axis,df['baseline_percent'],'--', label = new_label + " Baseline", color = "red")
		
		#plt.legend(loc="upper left", bbox_to_anchor=(1,1))


def see_data(double_dic, double_key, crowdsize = None):

	double_list = double_dic[double_key]

	df = pd.DataFrame(double_list, columns = ["true_pos", "true_neg", "false_pos", "false_neg", "total", "baseline"])

	df['percentage'] = (df.true_pos+ df.true_neg)/df.total

	df_short = df.loc[:,["true_pos", "true_neg", "false_pos", "false_neg", 'percentage',"total"]]

	if crowdsize == None:
		display(df_short)

	else:

		display(df_short.iloc[crowdsize - 1,:])

#custom ideology count
def count_ideo_func(count_ideo_dic, min_num = 5):
	
	ideo_count = {'l':0,'c':0,'m':0}
	
	for keys_it, values_it in count_ideo_dic.items():
		
		if keys_it < 0:
			
			ideo_count['l'] += values_it
		
		elif keys_it == 0:
			
			ideo_count['m'] += values_it
			
		elif keys_it > 0:
			
			ideo_count['c'] += values_it

	if ideo_count['l'] >= min_num and ideo_count['m'] >= min_num and ideo_count['c'] >= min_num:
		return True
	else:
		return False




