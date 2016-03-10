__author__ = 'Shi Fan'
import numpy as np, pandas as pd

def read(file=None):
	_df = pd.read_csv(file)
	_df.action_type.replace(np.nan, '-nan-', inplace=True)
	_df = _df.dropna(subset=['user_id']).reset_index(drop=True)
	return _df

def feature_extractor(_features=None, _df=None):
	df = pd.DataFrame(columns=_features)
	df.id = _df.user_id.unique()
	for i,j in df.id.iteritems():
		df.loc[i,'tot_time_elapsed'] = _df[_df.user_id==j].secs_elapsed.sum()
		df.loc[i,'avg_time_elapsed'] = _df[_df.user_id==j].secs_elapsed.sum()/len(_df[_df.user_id==j])
		df.loc[i,'most_freq_action_type'] = _df[_df.user_id==j].action_type.value_counts(dropna=False).idxmax(axis=1)
		df.loc[i,'most_time_action_type'] = (_df[_df.user_id==j].groupby('action_type').secs_elapsed.sum()/_df[_df.user_id==j].action_type.value_counts(dropna=False)).idxmax(axis=1)
		df.loc[i,'most_freq_device_type'] = _df[_df.user_id==j].device_type.value_counts(dropna=False).idxmax(axis=1)
		df.loc[i,'most_time_device_type'] = (_df[_df.user_id==j].groupby('device_type').secs_elapsed.sum()/_df[_df.user_id==j].device_type.value_counts(dropna=False)).idxmax(axis=1)
	return df

def main():
	sessions = read('../input/sessions.csv')

	features = ['id',
				'most_freq_action_type',
				'most_time_action_type',
				'most_freq_device_type',
				'most_time_device_type',
				'tot_time_elapsed',
				'avg_time_elapsed'
				]

	sessions_parsed = feature_extractor(features, sessions)
	sessions_parsed.to_csv('../output/sessions_parsed.csv')

if __name__=='__main__':
	main()