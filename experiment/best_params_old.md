#### MovieLens

----SVD----

Measure: RMSE
best score: 0.9587540720722535
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.01, 'reg_all': 0.04, 'n_epochs': 50}
Measure: MAE
best score: 0.7206774617398298
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.01, 'reg_all': 0.02, 'n_epochs': 50}

----UserItemTags----

Measure: RMSE
best score: 0.9332227134520406
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.01, 'reg_all': 0.04, 'n_epochs': 50}
Measure: MAE
best score: 0.6970535770781853
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.01, 'reg_all': 0.01, 'n_epochs': 50}

----ItemRelTags----

Measure: RMSE
best score: 0.9531200578260741
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.01, 'reg_all': 0.02, 'n_epochs': 50}
Measure: MAE
best score: 0.713182171936603
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.01, 'reg_all': 0.02, 'n_epochs': 50}

----ItemTopics----

Measure: RMSE
best score: 0.9396548256407872
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.005, 'reg_all': 0.01, 'n_epochs': 50, 'n_lda_iter': 2000, 'alpha': 0.04, 'eta': 0.02}
Measure: MAE
best score: 0.7020842663505821
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.01, 'reg_all': 0.01, 'n_epochs': 50, 'n_lda_iter': 2000, 'alpha': 0.02, 'eta': 0.02}

----CrossItemTopics----

Measure: RMSE
best score: 0.9402910158652589
best params: {'n_lda_iter': 2000, 'reg_all': 0.01, 'n_factors': 100, 'n_epochs': 50, 'biased': False, 'alpha': 0.02, 'eta': 0.02, 'lr_all': 0.005}
Measure: MAE
best score: 0.7030493729253707
best params: {'n_lda_iter': 2000, 'reg_all': 0.01, 'n_factors': 100, 'n_epochs': 50, 'biased': False, 'alpha': 0.02, 'eta': 0.02, 'lr_all': 0.005}



#### LibraryThings

----SVD----
Measure: RMSE
best score: 0.8690496883701944
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.02, 'reg_all': 0.01, 'n_epochs': 50}
Measure: MAE
best score: 0.6322646886471357
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.04, 'reg_all': 0.01, 'n_epochs': 50}

----UserItemTags----
Measure: RMSE
best score: 0.8563955202398654
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.04, 'reg_all': 0.01, 'n_epochs': 50}
Measure: MAE
best score: 0.6230477003416617
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.04, 'reg_all': 0.01, 'n_epochs': 50}

----ItemRelTags----
Measure: RMSE
best score: 0.8656053987086849
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.02, 'reg_all': 0.01, 'n_epochs': 50}
Measure: MAE
best score: 0.6297158021833245
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.04, 'reg_all': 0.01, 'n_epochs': 50}

----ItemTopics----
Measure: RMSE
best score: 0.8620831384797565
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.02, 'reg_all': 0.01, 'n_epochs': 50, 'n_lda_iter': 2000, 'alpha': 0.02, 'eta': 0.01}
Measure: MAE
best score: 0.6276285757622058
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.04, 'reg_all': 0.01, 'n_epochs': 50, 'n_lda_iter': 2000, 'alpha': 0.04, 'eta': 0.02}

----CrossItemTopics-lt----
Measure: RMSE
best score: 0.8620254354517846
best params: {'n_lda_iter': 2000, 'reg_all': 0.02, 'n_factors': 100, 'n_epochs': 50, 'biased': False, 'alpha': 0.01, 'eta': 0.01, 'lr_all': 0.02}

Measure: MAE
best score: 0.6295227219409412
best params: {'n_epochs': 50, 'alpha': 0.02, 'eta': 0.01, 'lr_all': 0.04, 'n_factors': 100, 'biased': False, 'n_lda_iter': 2000, 'reg_all': 0.01}





##### ItemTopicTest

The dump has been saved as file dumps/grid_search_result/ItemTopicsTest-search_best_perf-lt
Measure: RMSE
best score: 0.9377822635550924
best params: {'alpha': 0.02, 'reg_all': 0.02, 'biased': False, 'eta': 0.01, 'lr_all': 0.005, 'n_lda_iter': 2000, 'n_factors': 60, 'n_epochs': 50}
Measure: MAE
best score: 0.701812220790338
best params: {'alpha': 0.02, 'reg_all': 0.02, 'biased': False, 'eta': 0.01, 'lr_all': 0.005, 'n_lda_iter': 2000, 'n_factors': 60, 'n_epochs': 50}

Mean RMSE: 0.9401
Mean MAE : 0.7029

{'alpha': 0.02, 'reg_all': 0.01, 'biased': False, 'eta': 0.02, 'lr_all': 0.005, 'n_lda_iter': 2000, 'n_factors': 60, 'n_epochs': 50}

The dump has been saved as file dumps/grid_search_result/ItemTopicsTest-search_best_perf-lt
Measure: RMSE
best score: 0.8573677058087725
best params: {'reg_all': 0.02, 'lr_all': 0.02, 'eta': 0.02, 'alpha': 0.02, 'n_lda_iter': 2000, 'n_factors': 100, 'biased': False, 'n_epochs': 50}
Measure: MAE
best score: 0.6255696459702811
best params: {'reg_all': 0.01, 'lr_all': 0.1, 'eta': 0.01, 'alpha': 0.02, 'n_lda_iter': 2000, 'n_factors': 100, 'biased': False, 'n_epochs': 50}





----ItemTopics-lt----
Measure: RMSE
best score: 0.8622574402188553
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.08, 'reg_all': 0.01, 'n_epochs': 50, 'n_lda_iter': 6000, 'alpha': 0.04, 'eta': 0.04, 'n_topics': 10}
Measure: MAE
best score: 0.6253993802569037
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.08, 'reg_all': 0.005, 'n_epochs': 50, 'n_lda_iter': 6000, 'alpha': 0.02, 'eta': 0.01, 'n_topics': 10}

----CrossItemTopics-lt----
Measure: RMSE
best score: 0.8637137425350329
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.1, 'reg_all': 0.01, 'n_epochs': 50, 'n_lda_iter': 6000, 'alpha': 0.04, 'eta': 0.02, 'n_topics': 20}
Measure: MAE
best score: 0.6270999486611736
best params: {'biased': False, 'n_factors': 100, 'lr_all': 0.1, 'reg_all': 0.01, 'n_epochs': 50, 'n_lda_iter': 6000, 'alpha': 0.04, 'eta': 0.02, 'n_topics': 20}