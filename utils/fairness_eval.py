import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def make_dataframe(gender, y_true, y_pred, save=False):
    """pack the three series into a dataframe
    """
    pred_df = pd.DataFrame({'gender': gender,
                            'y_true': y_true,
                            'y_pred': y_pred})
    return pred_df

# def eval_fairness(gender, y_true, y_pred):
#     pred_df = make_dataframe(gender, y_true, y_pred)

#     confusion_mtx_male = confusion_matrix(pred_df.loc[pred_df['gender'] == 0, 'y_true'], pred_df.loc[pred_df['gender'] == 0, 'y_pred'])
#     confusion_mtx_male_normalized = confusion_mtx_male / confusion_mtx_male.sum(axis=1).reshape(-1, 1)
#     confusion_mtx_female = confusion_matrix(pred_df.loc[pred_df['gender'] == 1, 'y_true'], pred_df.loc[pred_df['gender'] == 1, 'y_pred'])
#     confusion_mtx_female_normalized = confusion_mtx_female / confusion_mtx_female.sum(axis=1).reshape(-1, 1)
    
#     m = confusion_mtx_male_normalized
#     f = confusion_mtx_female_normalized
#     equal_opportunities = np.array([np.min((m[i, i] / f[i, i], f[i, i] / m[i, i])) for i in range(len(m))])
#     print('["ang","sad","hap","neu"]')
#     print(equal_opportunities)

class FairnessEvaluation:

    def __init__(self, gender, y_true, y_pred):
        self.gender, self.y_true, self.y_pred = gender, y_true, y_pred
        self.pred_df = make_dataframe(self.gender, self.y_true, self.y_pred)
        self.confusion_mtx_male = confusion_matrix(self.pred_df.loc[self.pred_df['gender'] == 0, 'y_true'], self.pred_df.loc[self.pred_df['gender'] == 0, 'y_pred'])
        self.confusion_mtx_male_normalized = self.confusion_mtx_male / self.confusion_mtx_male.sum(axis=1).reshape(-1, 1)
        self.confusion_mtx_female = confusion_matrix(self.pred_df.loc[self.pred_df['gender'] == 1, 'y_true'], self.pred_df.loc[self.pred_df['gender'] == 1, 'y_pred'])
        self.confusion_mtx_female_normalized = self.confusion_mtx_female / self.confusion_mtx_female.sum(axis=1).reshape(-1, 1)
        
        m = self.confusion_mtx_male_normalized
        f = self.confusion_mtx_female_normalized
        self.equal_opportunities = np.array([np.min((m[i, i] / f[i, i], f[i, i] / m[i, i])) for i in range(len(m))])
        
    def print_equal_opportunities(self):
        print('["ang","sad","hap","neu"]')
        print(self.equal_opportunities)

