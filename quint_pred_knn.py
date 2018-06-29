import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
import seaborn as sns
from scipy import stats, integrate
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, validation_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


data = pd.read_csv('/Users/harsh/Desktop/quint/headline_data.csv', low_memory = False)
data.columns = ['content_id','content_publishing_day','content_time_to_read','author','hits_in_that_hour','content_category','content_tag','views','published_hour','Published_in_peak_hour','views_in_30_min','views_in_120_min','contentType','storyTemplate','readability_score','positive','negative','neutral','republished_count','events','organization','persons','locations','headline','slug','sims1Views','sims2Views','sims3Views','sims4Views','sims5Views','storyTemplate_code','agg_category_score','agg_tag_score','headline_sentiment', 'headline_wordcount']
data['avg_sims_views'] = data[['sims1Views','sims2Views','sims3Views','sims4Views','sims5Views']].mean(axis=1)

S2 = pd.read_csv('/Users/harsh/Desktop/QUINT_F/csv_files/text_sentiment.csv')
S2.columns = ['content_id','text_sentiment']
data = pd.merge(data,S2, on = 'content_id', how = 'left')
data['text_sentiment']=data['text_sentiment'].fillna(0)


data = data.drop_duplicates()
print(data.content_id.unique().shape)
print(data.shape)



data['views_in_30_min'].astype(float)
model_features = ['agg_category_score','republished_count','content_publishing_day','readability_score','agg_tag_score','storyTemplate_code','views_in_120_min','headline_sentiment','avg_sims_views','views']
#'events','organization','persons','locations'
#'content_publishing_day','content_time_to_read','published_hour'
#'hits_in_that_hour','views_in_30_min','views_in_120_min'





df = data[model_features]
df = df.drop_duplicates()
print (len(df))
print df.shape

labels = np.array(df['views'])
df = df.drop('views', axis = 1)
df = np.array(df)

def var_score(predicted_values, test_values):
    correct_labels = 0.0
    for i,val  in enumerate(test_values):
        digit_len = len(str(int(val)))-1
        abs_difference = abs(float(predicted_values[i]) - float(test_values[i]))
        if abs_difference <= 0.4*(float(test_values[i])):
            correct_labels += 1
            #print predicted_values[i], test_values[i], 1
        #else:
            #print predicted_values[i], test_values[i], 0

    return correct_labels/len(test_values)

train_data, test_data, train_labels, test_labels = train_test_split( df, labels, test_size=0.3, random_state=76)
#poly = PolynomialFeatures(degree=2)
#transform_train_data = poly.fit_transform(train_data)


parameters = {'n_neighbors':[5, 10, 15, 25, 30]}
knn = KNeighborsRegressor()
grid_obj = GridSearchCV(knn, parameters, cv=10, scoring='r2')
grid_obj.fit(train_data, train_labels)

print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)
print(pd.DataFrame(grid_obj.cv_results_))

param_range = tuple(list(range(1,20)))
train_scores, test_scores = validation_curve(KNeighborsRegressor(), train_data, train_labels, param_name="n_neighbors", param_range=param_range,cv=10)
test_scores_mean = np.mean(test_scores, axis=1)

knn = KNeighborsRegressor(n_neighbors=15)
knn.fit(train_data, train_labels)
predicted_knn = knn.predict(test_data)
vs = var_score(predicted_knn, test_labels)
print vs

#model = linear_model.LinearRegression()
#model.fit(train_data,train_labels)
#predictions = model.predict(test_data)
#print var_score(predictions, test_labels)

plt.scatter(predicted_knn, test_labels, alpha=0.5)
plt.plot(test_labels, test_labels)
plt.plot(test_labels, test_labels+0.4*test_labels)
plt.plot(test_labels, test_labels-0.4*test_labels)

plt.xlim(10000,60000)
plt.ylim(10000,60000)
plt.xlabel('predicted values')
plt.ylabel('actual values')
#plt.title('knn regressor ' + str(round((vs*100), 2) + ' accuracy')
plt.show()












'''Cs = [10, 100]
gammas = [0.001, 0.01, 0.1, 1]
parameters = {'C': Cs, 'gamma' : gammas}
grid_obj = GridSearchCV(SVR(kernel='linear'), parameters, cv=3)
grid_obj.fit(train_data, train_labels)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)'''







































'''model_dict = {

    'liRegression':linear_model.LinearRegression(),
    'xgb':XGBRegressor(),
    'huber' : linear_model.HuberRegressor(),
    'EN': ElasticNet(),
    'ridge': linear_model.Ridge(alpha = 0.1),
    'lasso': linear_model.LassoLars(alpha = 0.1)

}

results = {}
for i in range(50):
    train_data, test_data, train_labels, test_labels = train_test_split(df, labels, test_size=0.20, random_state=i)
    for key in model_dict:
            model = model_dict[key]
            model.fit(train_data,train_labels)
            predictions = model.predict(test_data)
            if key in results:
                results[key] += var_score(predictions, test_labels)
            else:
                results[key] = var_score(predictions, test_labels)

print results
results_avg = {

    'liRegression':results['liRegression']/50,
    'xgb':results['xgb']/50,
    'huber' : results['huber']/50,
    'EN': results['EN']/50,
    'ridge': results['ridge']/50,
    'lasso': results['lasso']/50


}
print results_avg'''
