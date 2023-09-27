# Boosting Overview
**What is Boosting?**
- Not a specific machine learning algorithm
- Concept that can be applied to a set of machine learning models
- "Meta-algorithm": Ensemble meta-algorithm used to convert many weak learners into a strong learner

**Weak learners and strong learners**
- Weak learner: ML algorithm that is slightly better than chance
- Boosting converts a collection of weak learners into a strong learner
- Strong learner: Any algorithm that can be tuned to achieve good performance.

**How boosting is accomplished?**
- Iteratively learning a set of week models on subsets of the data
- Weighting each weak prediction according to each weak learner's performance
- Combine the weighted predictions to obtain a single weighted prediction
- that is much better than the individual predictions themselves!

#### Gradient Boosting Types

- Gradient Boosted Trees https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
- Histogram-based Gradient Boosting Regression Tree: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor
- XG Boost (https://xgboost.readthedocs.io/en/stable/index.html)
- LightGBM (https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)
- CatBoost
- AdaBoost

## Parameters

**General Parameters**
- `booster` (default=gbtree) 

- `verbosity` (default=1): Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug)

- `early_stopping_rounds`: related to `num_boosting_round`. Instead of specificying a hard value of `num_boosting_round`, we use early stopping. If the hold-out  metric on a hold-out set does not change for the # of `early_stopping_rounds` then we stop optimization. If the holdout metric continuously improves up through when num_boost_rounds is reached, then early stopping does not occur.

**Parameters for Tree Booster**
- `n_estimators`: specifies how many sequential trees we want to make that attempt to correct for prior trees. Also, called Number of boosting rounds `num_boost_round`.

- `learning_rate` [default=0.3, eta]: associated with each tree/boosting round on how we calculate residual. Y = Y - alpha[k] * learner[k].predict(X) for the $k^{th}$ tree and learner is the tree we build. 

- `min_samples_split`: is the minimum weight (or number of samples if all samples have a weight of 1) required in order to create a new node in the tree. A smaller min_child_weight allows the algorithm to create children that correspond to fewer samples, thus allowing for more complex trees, but again, more likely to overfit. DOES NOT EXISTS IN SKLEARN API

- `max_depth` [default=6]: Depth of each tree. 

- `max_leaves`: maximum number of leaves, 0 indicates no limit

- `colsample_bytree`: fraction of features to choose from at every split in a given tree. Subsampling occurs once for every tree constructed. Others are `colsample_bylevel`, `colsample_bynode`.

- `min_child_weight`: The number of samples required to form a leaf node (the end of a branch). A leaf node is the termination of a branch and therefore the decision node of what class a sample belongs to.

- `subsample` [default=1]: corresponds to the fraction of observations (the rows) to subsample at each step. By default it is set to 1 meaning that we use all rows. Subsample occurs once in every boosting iteration. Subsample - 0.5 means xgboost would randomly sampple half training data prior to growing trees. 

- `scale_pos_weight`: Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative instances) / sum(positive instances)

- `sampling_method`: The method to use to sample the training instances. This has two values: `uniform`, `gradient_based` 

- `max_delta_step`

- `tree_method`: tree construction method. 

- `eval_metric`: evaluation metric for validation dataset, rmse for regression, logloss for classification. 

**Regularization in XG Boost (https://github.com/goodboychan/goodboychan.github.io/blob/main/_notebooks/2020-07-07-02-Fine-tuning-your-XGBoost-model.ipynb)**
- `gamma` [default=0, alias: min_split_loss:] minimum loss reduction to create a new tree split 

- `alpha`: L1 reg on leaf weights (For tree booster).,  L1 reg on weights (For linear booster). 

- `lambda`: L2 reg on leaf weights (For tree booster).,  L2 reg on weights. For linear booster. 


**Base Learners in XG boost**
- Linear Base Learners: Linear Regression with both L1 and L2 regularization (Elastic Net ) as base leaner instead of decision tree. 
- Tree Base Learners: the base learner is gbtree, short for “gradient boosted tree,” which operates like a standard sklearn Decision Tree.

**Learning Task Parameters**
- `objective`: `reg:squaredloss` regression with squared loss, `binary:logistic` logistic regression for binary classification, output is probability 
- `binary:logistic` is the default objective for XGBClassifier

#### References
- https://kevinvecmanis.io/machine%20learning/hyperparameter%20tuning/dataviz/python/2019/05/11/XGBoost-Tuning-Visual-Guide.html
- https://goodboychan.github.io/python/datacamp/machine_learning/2020/07/07/02-Fine-tuning-your-XGBoost-model.html
- https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster


