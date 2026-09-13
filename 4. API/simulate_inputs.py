import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import PowerTransformer
import joblib


def downtime_data():
    downtime = pd.read_csv('downtime_lease_nov22.csv', low_memory=False)
    downtime = downtime[(downtime['vacant_months'].notnull()) & (downtime['vacant_months'] > 0)]
    downtime = downtime[(downtime['year_off_market'] > 2007) &
                        (downtime['year_off_market'] < 2021)]
    downtime['building_rating_id'] = downtime['building_rating_id'].astype('str')
    downtime['cbsaid'] = downtime['cbsaid'].astype('str')
    downtime['age_of_building'] = datetime.datetime.now().year - downtime['construction_year']
    downtime = downtime[~((downtime['age_of_building'].isnull()) | (downtime['age_of_building'] < 0))]
    downtime['recession'] = downtime['year_on_market'].apply(lambda x: 'Y' if x in [2009, 2008] else 'N')
    downtime['month_on_market'] = pd.DatetimeIndex(downtime['date_on_market']).month.astype('str')
    downtime = downtime[~(downtime['estimated_rent'].isna())]

    cluster = pd.read_csv('cluster_submarket_Nov22.csv', low_memory=False)
    cluster = cluster[['cbsaid', 'submarket_name', 'Rent_bin', 'Rent']]
    cluster['cbsaid'] = cluster['cbsaid'].astype('str')
    downtime = pd.merge(downtime, cluster, left_on=['cbsaid', 'submarket_name'],
                        right_on=['cbsaid', 'submarket_name'], how='left')
    downtime['age_bldg_bins'] = downtime['construction_year'].apply(lambda x: 'New' if x > 2000 else
    ('Old' if x < 1960 else 'In-Between'))
    downtime['cbsa_cities'] = downtime.cbsa_cities.str.strip()
    downtime['cbsa_states'] = downtime.cbsa_states.str.strip()

    downtime['cbsa_cbsa_city'] = downtime['cbsaid'].astype('str') + "-" + downtime['cbsa_cities']
    downtime['cbsa_city_submarket'] = downtime['cbsa_cities'].astype('str') + " / " + downtime['submarket_name']

    return downtime, cluster

def rent(rent_min, rent_max, rent_orig, simulation):

    mu_log_rent = np.log(np.mean(rent_orig))
    sd_log_rent = np.log(np.std(rent_orig))

    rent_list = []
    while len(rent_list) < simulation:
        est_rent_lognormal = np.random.lognormal(mu_log_rent, sd_log_rent, size=None)
        if ((est_rent_lognormal < rent_min) | (est_rent_lognormal > rent_max)):
            continue
        else:
            rent_list.append(est_rent_lognormal)
    rent_df = pd.DataFrame(np.array(rent_list).reshape(-1, len(rent_list))).T
    rent_df.rename(columns={0: 'rent'}, inplace=True)

    return rent_df, mu_log_rent, sd_log_rent

def sqft(sqft_min, sqft_max, sqft_orig, simulation):
    sqft_list = []
    while len(sqft_list) < simulation:
        sqft_exp = np.random.exponential(scale=3131.900, size=None)
        if ((sqft_exp < sqft_min) | (sqft_exp > sqft_max)):
            continue
        else:
            sqft_list.append(sqft_exp)
    sqft_df = pd.DataFrame(np.array(sqft_list).reshape(-1, len(sqft_list))).T
    sqft_df.rename(columns={0: 'sqft_max'}, inplace=True)

    return sqft_df

def rating(building_rating_probs, simulation):
    building_rating_values = [1, 2, 3, 4, 5]
    building_rating_prob = building_rating_probs
    building_rating_discrete = np.random.choice(building_rating_values, simulation,
                                                p=building_rating_prob)
    building_rating_df = pd.DataFrame(building_rating_discrete, columns=['building_rating'])
    building_rating_df['building_rating'].value_counts()

    return building_rating_df

def age_of_building_func(age_of_building_prob, simulation):
    age_of_bldg = ['New',  'In-between', 'Old']

    age_of_building_discrete = np.random.choice(age_of_bldg, simulation, p=age_of_building_prob)
    age_of_building_df = pd.DataFrame(age_of_building_discrete, columns=['age_of_building'])
    return age_of_building_df

def month_on_market_func(month_market_prob, simulation):
    month_on_market_encode = [3, 6, 9, 12]
    month_on_market_discrete = np.random.choice(month_on_market_encode, simulation,  p=month_market_prob)
    month_on_market_df = pd.DataFrame(month_on_market_discrete, columns=['month_on_market'])

    return month_on_market_df


def cluster_func(cluster_probs, simulation):
    cluster_labels = ['High Quality Expensive',
                      'High Quality Less Expensive',
                      'Low Quality Expensive',
                      'Low Quality Less Expensive',
                      'Medium Quality Expensive',
                      'Medium Quality Less Expensive']

    cluster_probs = cluster_probs
    cluster_discrete = np.random.choice(cluster_labels, simulation, p=cluster_probs)
    cluster_df = pd.DataFrame(cluster_discrete, columns=['Rent_bin'])

    return cluster_df

def simulate(sim, rent_df, sqft_df, cluster_df, months_df, rating_df, age_df ):

    df = rent_df.join(sqft_df)
    df = df.join(rating_df)
    df = df.join(age_df)
    df = df.join(months_df)
    df = df.join(cluster_df)

    df['building_rating'] = df['building_rating'].astype('str')
    df['month_on_market'] = df['month_on_market'].astype('str')
    X_pred = df[['sqft_max',
                 'rent',
                 'building_rating',
                 'Rent_bin',
                 'month_on_market',
                 'age_of_building']]

    rf_tree = joblib.load(open('best_downtime_rftree.pkl', 'rb'))
    downtime_model_cols = joblib.load(open('downtime_cols.pkl', 'rb'))

    X_pred = pd.get_dummies(X_pred)
    X_pred = X_pred.reindex(columns=downtime_model_cols, fill_value=0)

    scalar = PowerTransformer()
    X_pred = scalar.fit_transform(X_pred)

    y_pred = rf_tree.predict(X_pred)
    y_pred = np.round(10**y_pred, 0)

    return y_pred
