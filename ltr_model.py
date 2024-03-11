import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score
train = pd.read_csv('train_df.csv')
test = pd.read_csv('test_df.csv')

X_train = train[train.columns[1:-1]]
y_train = train[train.columns[-1]]
X_test = test[test.columns[1:-1]]
y_test = test[test.columns[-1]]



gr = train.groupby(by=train['search_id'])
size = gr.size()
group_train = size.to_list()
gr = test.groupby(by=test['search_id'])
size = gr.size()
group_test = size.to_list()

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
train_data.set_group(group_train)
test_data.set_group(group_test)

param = {
    "task": "train",
    "num_leaves": 30,
    "min_data_in_leaf": 1,
    "min_sum_hessian_in_leaf": 100,
    "objective": "lambdarank",
    "metric": "ndcg",
    "learning_rate": .01,
    "num_threads": 6,
    'boosting_type': 'gbdt',
    'colsample_bytree': 0.2

}

ltr = lgb.train(
    param,
    train_data,
    valid_sets=[test_data],
    valid_names=["test"],
    callbacks=[lgb.early_stopping(stopping_rounds=5), lgb.log_evaluation(1)])

test_pred = ltr.predict(X_test)
ndcg = ndcg_score([y_test], [test_pred])
print("\n NDCG score:", ndcg)
