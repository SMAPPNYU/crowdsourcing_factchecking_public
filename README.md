## Paper Abstract
Reducing the spread of false and misleading news remains a challenge for social media platforms, as the current strategy of using third-party fact-checkers lacks the capacity to address both the scale and speed of misinformation diffusion. Recent research on the “wisdom of the crowds” suggests one possible solution: aggregating the evaluations of groups of ordinary users to assess the veracity of online information. Using a pre-registered research design, we investigate the effectiveness of crowdsourced fact checking in real time.  We select popular news stories in real time and have them evaluated by both ordinary individuals and professional fact checkers within 72 hours of publication. Our data consists of 21,531 individual evaluations across 135 articles published between November 2019 and June 2020.  Although we find that machine learning based models (that use the crowd as input) perform significantly better than simple aggregation rules at identifying false news, our results suggest that neither approach is able to perform at the level of professional fact checkers. Additionally, both simple crowd aggregations and machine learning models perform best when only using the evaluations from survey respondents with high political knowledge, thus suggesting reason for caution for crowd sourcing models that seek to rely on a representative sample of the population. Overall, our analyses reveal that our crowd-based identification systems do provide some information on news quality, but are nonetheless limited in their ability to identify false news.

Bitex citation:

@unpublished{moderatingwmob,
  title={Moderating with the Mob: Evaluating the Efficacy of Real Time Crowdsourced Fact Checking},
  author={Godel, William and Sanderson, Zeve and Aslett, Kevin and Nagler, Jonathan, and Persily, Nathanial and Tucker, Joshua A. and Bonneau, Richard},
institution = {CSMaP},
  year={2021},
}



## Folders
- `source_data\` - all original source data for this project
- `prepared_data\` - converted crowds, based on original source data
- `heuristic_data\` - the performance of heuristic based methods (based on prepared data)
- `models\` - All ML models trained on prepared data
- `data_pickles\` - all the data that is saved as pickles that are not models, mostly intermediate data used for processing
- `code\` - all the project code


The structure of the repo is as follows:

Source data stores all the data that is the source for this project

This data is then processed so that it produces crowds. The code that does this is located in the "code" folder and is called:
- `crowd_source_data_preperation.ipynb` (original data)
- `crowd_source_data_preperation_covid.ipynb` (new covid data)
- `crowd_source_data_preperation_noncovid.ipynb` (new noncovid data)

**NOTE: results will be sensitive to the TRAIN/TEST split - Each of these files splits articles into either the train or test set. Splitting the articles any other way could alter our results, potentially significantly. I would suggest replicating using the same training and test split already used, which are saved in data_pickles.**

All the data created in these files is saved in the `prepared_data/` folder.

The only exception to this is that special crowds (those with high political knowledge) must be created based on this. These special crowds are created in `special_crowds.ipynb` - creates data for high CRT  and saves it to `data_pickles/`.

Following that this data is used to asses the performance of heuristic rules, the code that determines this performance is:
- `heuristics.ipynb` (heuristic analysis on all data)
- `heuristics_test.iypnb` (heuristic analysis on test data)
- `heuristics_test_high_pol.ipynb` (heuristic analysis on high political knowledge data)

All the data created by this is saved in `heuristic_data/`.

All graphs and heuristic results were calculated, created and explored in `graph_explore.ipynb`.

ML evaluation of crowds takes the same prepared data and runs ML on it. The files that do this are:
- `ML.ipynb` - this runs training on algorithms that had inputs of crowd size 10. 
- `ML_largecrowd.ipynb` - this runs training for algorithms that had inputs of crowd size 25

And all models from this are saved in `models/`.

Then all ML results on test are calculated in `Results_updated.ipynb`.

Analysis of feature importance is performed in `feature_importance.ipynb`.

We also performed a robustness check with different distributions of data:
- `test_alternatives.ipynb` - this file creates these distributions
- `robustness_checks.ipynb` - applies the ML algorithms to these distributions

## Erratum

This code was updated in February 2022 to reflect a correction to the file results_updated. The original version of results_updated is now saved as results_updated_original. The original version of results updated incorrectly reversed the calculation for false positives and false negatives.


