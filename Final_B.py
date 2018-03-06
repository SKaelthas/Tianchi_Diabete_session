# encoding: utf-8

"""
@Version: python3.6.2
@Py version: python3.6.2
@Author: sKaelthas
@License: Apache Licence 2.0
@Contact: ao.li@scrcnet.org
@Software: PyCharm
@File: Sample.py
@Time: 2018/1/16 


Copyright 2018 ao.li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn import linear_model
import sklearn.preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from dateutil.parser import parse
import re
import xgboost as xgb

data_path = './'
train = pd.read_csv(data_path + 'd_train_withA_20180129.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'd_test_B_20180128.csv', encoding='gb2312')


# answer_A = pd.read_csv(data_path + 'd_answer_a_20180128.csv',header=None, encoding='gb2312')


def make_feat(train, test):
    def rm_feature(input_data, rm_list):
        for rm_fea in rm_list:
            if rm_fea in data.columns.tolist():
                input_data = input_data.drop(rm_fea, 1)
        return input_data

    def feature_eng(input_data):
        mapper_df = DataFrameMapper([
            (["白细胞计数", "红细胞计数", "血红蛋白", "红细胞压积", "红细胞平均体积", "红细胞平均血红蛋白量",
              "红细胞平均血红蛋白浓度", "红细胞体积分布宽度",
              "血小板计数", "血小板平均体积", "血小板体积分布宽度", "血小板比积"], PCA(1)),
            (["甘油三酯"], sklearn.preprocessing.MinMaxScaler()),
        ], df_out=True, default=None)
        output_data = mapper_df.fit_transform(input_data.copy())
        return output_data

    def add_fea(data, fea_file):

        fea_level1 = pd.Series.from_csv(fea_file, encoding='gb2312').sort_values()
        fea_level1_50 = fea_level1.head(40)
        for fea in fea_level1_50.index:
            fea = fea.replace("*r-", "*r_")
            fea_reg = re.split('([\+\-x\/\^])', fea)
            expr = "data[\"" + str(fea) + "\"]="
            for idx in range(len(fea_reg)):
                if idx % 2 == 0:
                    expr += ("data[\"" + fea_reg[idx] + "\"]")
                else:
                    if fea_reg[idx] == "x":
                        expr += "*"
                    elif fea_reg[idx] == "^":
                        expr += "**"
                    else:
                        expr += fea_reg[idx]
            # print(expr)
            expr = expr.replace("*r_", "*r-")
            if re.search(r'\^', fea):
                # print(fea)
                expr = expr.replace("]=", "]=(")
                expr += ").apply(np.log)"
                # print(expr)
            exec(expr)
        return data

    def add_fea_level2(data):
        ext = []
        check_list = data.columns.tolist().copy()
        for fea1 in check_list:
            if fea1 != "id" and fea1 != "血糖":
                ext.append(fea1)
                print(ext)
                for fea2 in check_list:
                    if fea1 != fea2 and (fea2 not in ext) and fea2 != "id" and fea2 != "血糖":
                        # print(fea2)
                        data[fea1 + ")+(" + fea2] = data[fea1] + data[fea2]
                        data[fea1 + ")-(" + fea2] = data[fea1] - data[fea2]
                        data[fea1 + ")x(" + fea2] = data[fea1] * data[fea2]
                        data[fea1 + ")/(" + fea2] = data[fea1] / data[fea2]
        return data

    def add_poly_fea(data, fea_file):

        fea_level1 = pd.Series.from_csv(fea_file, encoding='gb2312').sort_values()
        fea_level1_50 = fea_level1.head(200)
        for fea in fea_level1_50.index:
            fea = fea.replace("*r-", "*r_")
            fea = fea.replace(" ", "x")
            fea_reg = re.split('([\+\-x\/\^])', fea)
            expr = "data[\"" + str(fea) + "\"]="
            for idx in range(len(fea_reg)):
                if idx % 2 == 0:
                    if fea_reg[idx] == "2":
                        continue
                    else:
                        expr += ("data[\"" + fea_reg[idx] + "\"]")
                else:
                    if fea_reg[idx] == "x":
                        expr += "*"
                    elif fea_reg[idx] == "^":
                        expr += "*"
                        expr += ("data[\"" + fea_reg[idx - 1] + "\"]")
                    else:
                        expr += fea_reg[idx]
            # print(expr)
            expr = expr.replace("*r_", "*r-")
            # print(expr)
            exec(expr)
        return data

    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train, test])
    data['性别'] = data['性别'].map({'男': 1, '女': 2})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-09-01')).dt.days
    data.fillna(data.median(axis=0), inplace=True)
    fea_file = "fea_poly_LR.csv"
    data = add_poly_fea(data, fea_file)
    #print(data.columns.tolist())
    #print(data.shape)
    fea_file = "fea_level1.csv"
    data = add_fea(data, fea_file)

    RM_LIST = ["*总蛋白", "白细胞计数", "红细胞压积", "白蛋白", "乙肝表面抗原", "乙肝表面抗体", "乙肝e抗原", "乙肝e抗体",
               "单核细胞%", "嗜酸细胞%", "总胆固醇", "高密度脂蛋白胆固醇", "肌酐", ]
    rm_list = ["性别", "年龄", "*天门冬氨酸氨基转换酶", "*丙氨酸氨基转换酶", "*碱性磷酸酶", "*r-谷氨酰基转换酶", "*总蛋白", "白蛋白", "*球蛋白", "甘油三酯", "总胆固醇",
               "高密度脂蛋白胆固醇", "低密度脂蛋白胆固醇", "尿素", "肌酐", "尿酸", "乙肝表面抗原", "乙肝表面抗体", "乙肝e抗原", "乙肝e抗体", "乙肝核心抗体", "白细胞计数",
               "红细胞计数", "血红蛋白", "红细胞压积", "红细胞平均体积", "红细胞平均血红蛋白量", "红细胞平均血红蛋白浓度", "红细胞体积分布宽度", "血小板计数", "血小板平均体积",
               "血小板体积分布宽度", "血小板比积", "中性粒细胞%", "淋巴细胞%", "单核细胞%", "嗜酸细胞%", "嗜碱细胞%"]
    data = rm_feature(data, RM_LIST)
    # data = add_fea_level2(data)
    # data = feature_eng(data)
    #print(data.columns.tolist())
    #print(data.shape)
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    # train_feat.to_csv(r'train_tmp.csv', index=False, float_format='%.4f')
    # test_feat.to_csv(r'test_tmp.csv', index=False, float_format='%.4f')
    train_feat = train_feat.drop("id", 1)
    test_feat = test_feat.drop("id", 1)
    print(train_feat.columns.tolist())
    print(train_feat.shape)
    return train_feat, test_feat

    # for fea1 in rm_list:
    #   for fea2 in rm_list:
    #          data[fea1 + "^" + fea2] = data[fea1] ** data[fea2]


train_feat, test_feat = make_feat(train, test)
predictors = [f for f in test_feat.columns if f not in ['血糖']]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('0.5mse', score, False)


# Parameters
FUDGE_FACTOR = 1  # Multiply forecasts by this
XGB_WEIGHT = 0.500
BASELINE_WEIGHT = 0.0100
OLS_WEIGHT = 0.080
NN_WEIGHT = 0.150
CAT_WEIGHT = 0.0000
# XGB1_WEIGHT = 0.8000  # Weight of first in combination of two XGB models
BASELINE_PRED = 5.631925  # Baseline based on mean of training data, per Oleg

print('开始训练...')

# print('开始CV 10训练...')

k = 10
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], k))
kf = KFold(len(train_feat), n_folds=k, shuffle=True, random_state=1108)
for i, (train_index, test_index) in enumerate(kf):
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]

    ##   EN     ##
    x_train = train_feat1[predictors]
    y_train = train_feat1['血糖']
    x_test = train_feat2[predictors]
    y_test = train_feat2['血糖']
    reg = linear_model.ElasticNet(alpha=0.01, l1_ratio=0.3)
    reg.fit(x_train, y_train)
    y_pred_en = reg.predict(x_test)
    y_result_en = reg.predict(test_feat[predictors])
    # print(y_result_en)

    ######LGB
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'], categorical_feature=['性别'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    params = {
        'learning_rate': 0.005,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        # 'sub_feature': 0.5,
        'num_leaves': 7,
        # 'colsample_bytree': 0.7,
        'feature_fraction': 0.5,
        'min_data': 60,
        'min_hessian': 1,
        'verbose': -1,
    }
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=100000,
                    valid_sets=lgb_train2,
                    verbose_eval=1000,
                    feval=evalerror,
                    early_stopping_rounds=1000)
    if i == 0:
        feat_imp_sum = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    else:
        feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
        feat_imp_sum += feat_imp
    lgb_pre = gbm.predict(train_feat2[predictors])
    lgb_result = gbm.predict(test_feat[predictors])
    print("Type of nn_pred is ", type(train_preds[test_index]))
    print("Shape of nn_pred is ", train_preds[test_index].shape)

    #### XGB
    xgb_params = {
        'eta': 0.025,
        'max_depth': 6,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'lambda': 0.8,
        'alpha': 0.9,
        # 'base_score': BASELINE_PRED,
        # 'silent': 1,
        'seed': 1108,
        'early_stopping_rounds': 5
    }
    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)
    num_boost_rounds = 200
    print("num_boost_rounds=" + str(num_boost_rounds))
    # train model
    print("Training XGBoost ...")
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
    xgb_pred = model.predict(dtest)
    dresult = xgb.DMatrix(test_feat[predictors])
    xgb_result = model.predict(dresult)
    # print(xgb_result)

    ###NN
    ## Preprocessing
    x_train = train_feat1[predictors]
    y_train = train_feat1['血糖']
    x_test = train_feat2[predictors]
    y_test = train_feat2['血糖']
    imputer = Imputer()
    imputer.fit(x_train.iloc[:, :])
    x_train = imputer.transform(x_train.iloc[:, :])
    imputer.fit(x_test.iloc[:, :])
    x_test = imputer.transform(x_test.iloc[:, :])
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    len_x = int(x_train.shape[1])
    # Neural Network
    nn = Sequential()
    nn.add(Dense(units=200, kernel_initializer='normal', input_dim=len_x))
    nn.add(PReLU())
    nn.add(Dropout(.8))
    nn.add(Dense(units=80, kernel_initializer='normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.8))
    nn.add(Dense(units=16, kernel_initializer='normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.8))
    nn.add(Dense(1, kernel_initializer='normal'))
    nn.compile(loss='mse', optimizer=Adam(lr=4e-3, decay=1e-4))
    nn.fit(np.array(x_train), np.array(y_train), batch_size=80, epochs=100, verbose=2)
    print("Predicting with neural network model...")
    # print("x_test.shape:",x_test.shape)
    y_pred_ann = nn.predict(x_test)
    nn_pred = y_pred_ann.flatten()
    # print(pd.DataFrame(nn_pred).head())
    y_result_nn = nn.predict(sc.transform(test_feat[predictors])).flatten()
    # print(y_result_nn)
    # train_preds[test_index] += nn.predict(x_test)
    # test_preds[:, i] = nn.predict(test_feat[predictors])

    ### catboost
    """
    exclude_unique = []
    num_ensembles=5
    y_pred_cat = 0.0
    for c in train_feat.columns:
        num_uniques = len(train_feat[c].unique())
        if train_feat[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1:
            exclude_unique.append(c)
    print("We exclude: %s" % len(exclude_unique))
    print("Define training features !!")
    exclude_other = ['id', '血糖', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']
    train_features = []
    for c in train_feat.columns:
        print(c)
        train_features.append(c)
    print("We use these for training: %s" % len(train_features))
    print("Define categorial features !!")
    cat_feature_inds = []
    cat_unique_thresh = 10
    for i, c in enumerate(train_features):
        num_uniques = len(train_feat[c].unique())
        if num_uniques < cat_unique_thresh:
            cat_feature_inds.append(i)
    for i in tqdm(range(num_ensembles)):
        model = CatBoostRegressor(
            iterations=1000, learning_rate=0.03,
            depth=6, l2_leaf_reg=3,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=i)
        model.fit(
            x_train, y_train,
            cat_features=cat_feature_inds)
        y_pred_cat += model.predict(x_test)
    y_pred_cat /= num_ensembles
    """

    ##combine
    lgb_weight = 1 - BASELINE_WEIGHT - NN_WEIGHT - OLS_WEIGHT - XGB_WEIGHT
    lgb_weight0 = lgb_weight / (1 - OLS_WEIGHT)
    xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
    cat_weight0 = CAT_WEIGHT / (1 - OLS_WEIGHT)
    baseline_weight0 = BASELINE_WEIGHT / (1 - OLS_WEIGHT)
    nn_weight0 = NN_WEIGHT / (1 - OLS_WEIGHT)
    pred0 = 0
    pred0 += baseline_weight0 * BASELINE_PRED
    pred0 += lgb_weight0 * lgb_pre
    pred0 += xgb_weight0 * xgb_pred
    pred0 += nn_weight0 * nn_pred
    pred = FUDGE_FACTOR * (OLS_WEIGHT * y_pred_en + (1 - OLS_WEIGHT) * pred0)
    train_preds[test_index] += pred

    result0 = 0
    result0 += baseline_weight0 * BASELINE_PRED
    result0 += lgb_weight0 * lgb_result
    result0 += xgb_weight0 * xgb_result
    result0 += nn_weight0 * y_result_nn
    result = FUDGE_FACTOR * (OLS_WEIGHT * y_result_en + (1 - OLS_WEIGHT) * result0)
    test_preds[:, i] = result
    # print('单次A榜得分：{:.4f}'.format(mean_squared_error(answer_A[0:], test_preds[0:, i]) * 0.5))
    # print(pred)

# print(train_preds)
print('线下得分：{:.4f}'.format(mean_squared_error(train_feat['血糖'], train_preds) * 0.5))
print('CV训练用时{:.1f}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
# print('A榜得分：{:.4f}'.format(mean_squared_error(answer_A[0:], submission[0:]) * 0.5))


submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False,
                  float_format='%.4f')
# feat_imp_sum.to_csv(r'fea_merge_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))


"""
("id", None),
("性别", None),
("年龄", None),
("体检日期", None),
("*天门冬氨酸氨基转换酶", None),
("*丙氨酸氨基转换酶", None),
("*碱性磷酸酶", None),
("*r-谷氨酰基转换酶", None),
("*总蛋白", None),
("白蛋白", None),
("*球蛋白", None),
("白球比例", None),
("甘油三酯", None),
("总胆固醇", None),
("高密度脂蛋白胆固醇", None),
("低密度脂蛋白胆固醇", None),
("尿素", None),
("肌酐", None),
("尿酸", None),
("乙肝表面抗原", None),
("乙肝表面抗体", None),
("乙肝e抗原", None),
("乙肝e抗体", None),
("乙肝核心抗体", None),
("白细胞计数", None),
("红细胞计数", None),
("血红蛋白", None),
("红细胞压积", None),
("红细胞平均体积", None),
("红细胞平均血红蛋白量", None),
("红细胞平均血红蛋白浓度", None),
("红细胞体积分布宽度", None),
("血小板计数", None),
("血小板平均体积", None),
("血小板体积分布宽度", None),
("血小板比积", None),
("中性粒细胞%", None),
("淋巴细胞%", None),
("单核细胞%", None),
("嗜酸细胞%", None),
("嗜碱细胞%", None),
("血糖", None),
"""
