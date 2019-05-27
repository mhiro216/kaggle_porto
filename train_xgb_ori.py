import pandas as pd
import numpy as np
from tqdm import tqdm
from logging import StreamHandler, DEBUG, INFO, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
import xgboost as xgb

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'

def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', -gini(y, pred)

if __name__ == '__main__':
    
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)
    
    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    
    logger.info('start')
    
    df = load_train_data()
    
    x_train = df.drop('target', axis=1)
    y_train = df['target'].values
    
    use_cols = x_train.columns.values
    
    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
    
    logger.info('data preparation end {}'.format(x_train.shape))
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    all_params = {'max_depth': [3, 5, 7], # アンサンブルは１つ１つの学習木の説明力はあまり高くない方が精度が良い。多くても10が目安
                  'learning_rate': [0.1], # コンペ最終盤になったら0.01とかにしてよりたくさんの木で作ると精度が上がること多し。ただ予測機を作るのに時間がかかるようになるので最終盤で良い。最初はfeature engineeringに集中
                  'min_child_weight': [3, 5, 10], # 大きいほど汎化性能up。もちろん精度とのトレードオフ
                  'n_estimators': [10000], # early_stopping入っているので大きくしておいても勝手に止まる？
                  'colsample_bytree': [0.8, 0.9], # 学習木１つ作るときどの割合の特徴を使うか。1だと全部。ある程度選んだ方が良い。0.5以上の値をサーチするのが推奨
                  'colsample_bylevel': [0.8, 0.9], # ツリーだけでなく学習の木の深さによっても使う特徴を減らす
                  "reg_alpha": [0, 0.1], # 正則化項。スプリットが発生しにくくなる。0.1で効くなら他の値も試す
                  'max_delta_step': [0.1],
                  'seed': [0]
                 }
    min_score = 100
    min_params = None
    
    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))
        
        list_logloss_score = []
        list_gini_score = []
        list_best_iterations = []
        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx,:]
            val_x = x_train.iloc[valid_idx,:]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            clf = xgb.sklearn.XGBClassifier(**params)
            clf.fit(trn_x, 
                    trn_y,
                    eval_set=[(val_x,val_y)],
                    early_stopping_rounds=10,
                    eval_metric=gini_xgb
                   )
            pred = clf.predict_proba(val_x, ntree_limit=clf.best_ntree_limit)[:, 1]
            sc_logloss = log_loss(val_y, pred)
            sc_gini =  - gini(val_y, pred)

            list_logloss_score.append(sc_logloss)
            list_gini_score.append(sc_gini)
            list_best_iterations.append(clf.best_iteration)
            logger.debug('    logloss: {}, gini: {}'.format(sc_logloss, sc_gini))
            break # hold-outでやる場合
        
        params['n_estimators'] = int(np.mean(list_best_iterations))
        sc_logloss = np.mean(list_logloss_score)
        sc_gini = np.mean(list_gini_score)
        if min_score > sc_gini:
            min_score = sc_gini
            min_params = params
        logger.info('logloss: {}, gini: {}'.format(sc_logloss, sc_gini))
        logger.info('current min score: {}, params: {}'.format(min_score, min_params))
    
    logger.info('mininum params: {}'.format(min_params))
    logger.info('mininum gini: {}'.format(min_score))

    clf = xgb.sklearn.XGBClassifier(**min_params)
    clf.fit(x_train, y_train)
    
    logger.info('train end')
    
    df = load_test_data()
    
    x_test = df[use_cols].sort_values('id')
    
    logger.info('test data load end {}'.format(x_test.shape))
    pred_test = clf.predict_proba(x_test)[:, 1]
    
    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test
    
    df_submit.to_csv(DIR + 'submit.csv', index=False)
    
    logger.info('end')

