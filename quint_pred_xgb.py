import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('/Users/harsh/Desktop/quint/headline_data.csv')
data.columns = ['content_id','content_publishing_day','content_time_to_read','author','hits_in_that_hour','content_category','content_tag','views','published_hour','Published_in_peak_hour','views_in_30_min','views_in_120_min','contentType','storyTemplate','readability_score','positive','negative','neutral','republished_count','events','organization','persons','locations','headline','slug','sims1Views','sims2Views','sims3Views','sims4Views','sims5Views','storyTemplate_code','avg_sims_views','agg_category_score','agg_tag_score','headline_sentiment']

S2 = pd.read_csv('/Users/harsh/Desktop/QUINT_F/csv_files/text_sentiment.csv')
S2.columns = ['content_id','text_sentiment']
data = pd.merge(data,S2, on = 'content_id', how = 'left')
data['text_sentiment']=data['text_sentiment'].fillna(0)

C1 = pd.read_csv('/Users/harsh/Desktop/QUINT_F/csv_files/processed_text_with_clustering.csv')
C1.columns = ['content_id','category','text','cluster']
data = pd.merge(data,C1, on = 'content_id',how = 'left')


data = data.drop_duplicates()
print(data.content_id.unique().shape)
print(data.shape)

data = data.dropna()


for i, val in enumerate(list(data['content_publishing_day'])):
    if val == 'a693fe38-effa-44c0-a9c1-4485c0aa3a97':
        print i


data['views_in_30_min'].astype(float)
model_features = ['views','content_publishing_day','agg_category_score','agg_tag_score','storyTemplate_code','headline_sentiment','views_in_120_min','avg_sims_views','views_in_30_min','readability_score','cluster','text_sentiment']
df = data[model_features]
df = df.drop_duplicates()
print (len(df))
print df.shape

labels = np.array(df['views'])
df = df.drop('views', axis = 1)
df = np.array(df)

train_data, test_data, train_labels, test_labels = train_test_split( df, labels, test_size=0.33, random_state=41)

model2 = linear_model.TheilSenRegressor()
model2.fit(train_data,train_labels)
predictions = model2.predict(test_data)

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
    return correct_labels/len(predicted_values)



#predictions.sort()
#test_labels.sort()
a = var_score(predictions, test_labels)
print a

plt.scatter(predictions, test_labels, alpha=0.5)
plt.plot(test_labels, test_labels)
plt.plot(test_labels, test_labels+0.4*test_labels)
plt.plot(test_labels, test_labels-0.4*test_labels)

plt.xlim(1000,60000)
plt.ylim(1000,60000)
plt.xlabel('predicted values')
plt.ylabel('actual values')
plt.title('TheilSenRegressor ' str(round((vs*100), 2)  ' accuracy')
plt.show()


'''prediction = pd.DataFrame(predictions, columns=['predictions'])
prediction['test_views'] = test_labels
prediction.to_csv("result1.csv")

plt.plot(predictions)
plt.plot(test_labels)
plt.show()
print(a)
print(a)'''

