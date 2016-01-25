
#!/usr/bin/python
from __future__ import division
import numpy as np
import pandas as pd
from regressor import Regressor
from sklearn.cross_validation import cross_val_score
from sklearn.utils import check_array
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from scipy import interp, arange, exp
from scipy.interpolate import interp1d,NearestNDInterpolator
from decimal import Decimal

nnanLimit=5
validData=None

def scoring_function(y_true, y_pred):
    y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def missingValuesHandler(row):
    nnans=row.isnull().sum()


    if(nnans<nnanLimit and nnans>0):
        #y_array=np.array(row.drop('ID').drop('date').drop('product_id').values)
        #x_array=np.arange(start=0,stop=len(y_array),step=1)
        #f_i = interp1d(x_array, y_array,fill_value=np.nan)
        #print np.isnan(y_array).sum()
        #print np.isnan(f_i(x_array)).sum()
        #inter=f_i(x_array)
        #plt.plot(x_array,y_array,'r--',x_array,inter,'o')
        #plt.show()
        #TODO average prior and post values
        row.fillna(method="ffill",inplace=True)
        row.fillna(method="bfill",inplace=True)



    elif nnans>0:
        #TODO pass dict with imputed values
        id=row['ID']
        date=row['date']
        product_id=row['product_id']
        val=validData.loc[product_id]
        row.fillna(value=val,inplace=True)
    return row


print "Loading test set"

testing_input=pd.read_csv("data/testing_input.csv")
testing_input=testing_input.set_index(['ID', 'date', 'product_id'])

print "Loading train set"
training_input=pd.read_csv("data/training_input.csv")
training_input=training_input.set_index(['ID', 'date', 'product_id'])

training_valid_data=training_input.copy().dropna(axis=0, how='any', subset=None, inplace=False)
testing_valid_data=testing_input.copy().dropna(axis=0, how='any', subset=None, inplace=False)
validData=pd.concat([training_valid_data, testing_valid_data],join='inner')
validData=validData.reset_index()

validData=validData.drop('ID',axis=1).drop('date',axis=1)
validData=(validData.groupby('product_id').mean())
validData=validData.mean(axis=1)
print "Loading train targets"

targets=pd.read_csv("data/challenge_output_data_training_file_prediction_of_transaction_volumes_in_financial_markets.csv", sep=';')
targets=targets.set_index(['ID'])
data=training_input.join(targets)
y=data['TARGET'].values
data=data.drop('TARGET',axis=1)
print "Processing"
data=data.reset_index().apply(missingValuesHandler, axis=1)
print "Finished processing"
data.to_csv("ImputedData.csv")
X=data.values
assert len(X) == len(y)
reg=Regressor().getRegressor()
print "Cross-Validation"
loss = make_scorer(scoring_function, greater_is_better=False)
score = cross_val_score(reg, X, y,cv=5,scoring=loss).mean()
print("Score = %.2f" % score)

print "Calculating Predictions"
#testing_input=testing_input.apply(missingValuesHandler,axis=1)
#predictions=reg.predict(testing_input)
