import pandas as pd
data = pd.read_csv("/Users/harsh/Desktop/headline_data/merged_with_headline_data.csv", error_bad_lines = False, header  = None )
data = data.iloc[:, :-1]
data.columns = ['content_id','content_publishing_day','content_time_to_read','author','hits_in_that_hour','content_category','content_tag','views','published_hour','Published_in_peak_hour','views_in_30_min','views_in_120_min','contentType','storyTemplate','readability_score','positive','negative','neutral','republished_count','events','organization','persons','locations','headLine','slug','sims1Views','sims2Views','sims3Views','sims4Views','sims5Views']
data['storyTemplate_code']=data['storyTemplate'].astype('category').cat.codes
W1 = pd.read_csv("/Users/harsh/Desktop/cbworkbooks/W1.csv")
W2 = pd.read_csv("/Users/harsh/130915/W5.csv")
data = pd.merge(data, W1, how='left', on='content_category')
data = pd.merge(data, W2, how='left', on='content_tag')

data['avg_sims_views'] = data[['sims1Views','sims2Views','sims3Views','sims4Views','sims5Views']].mean(axis=1)
data = data.fillna(method='ffill')
#print(data.shape)

data['agg_category_score'] = data.groupby(['content_id'])['score'].transform('sum')
data['agg_tag_score'] = data.groupby(['content_id'])['content_tag_score'].transform('sum')
del data['content_tag_score']
del data['score']

#data.drop_duplicates()
#print(data.content_id.unique().shape)

#print (len(data))
#print (data['views'][416194])
bad_indices=[]
for i in range(len(data)):
    if data['views'][i] == 'views':
        bad_indices.append(i)
#print(len(bad_indices))
#print(bad_indices)
for val in reversed(bad_indices):
    #print val
    data = data.drop(data.index[val])
    #print(len(data))
print(data[:8])
df = data['headLine']
data.to_csv("/Users/harsh/Desktop/headline_data.csv")
print data['headLine'][:5]
#data['headLine'] = data['headLine'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#print data['headLine'][:5]

