import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import simulate_inputs as sm


@st.cache(persist=True)
def load_data():
    data_df, cluster = sm.downtime_data()
    return data_df, cluster

@st.cache(persist=True)
def renewal_data():
    renewal = pd.read_csv('renewal_rate_nov22.csv', low_memory=False)
    renewal = renewal[(renewal['new_renewal_from_year'] > 2007) & (renewal['new_renewal_from_year'] < 2021)]
    cluster = pd.read_csv('cluster_submarket_Nov22.csv', low_memory=False)

    renewal = pd.merge(renewal,
                      cluster[['cbsaid', 'submarket_name', 'Rent_bin']],
                      left_on=['cbsaid', 'submarket_name'],
                      right_on=['cbsaid', 'submarket_name'], how='left')
    return renewal

def downtime_model(X_pred):
    #model = joblib.load(open('best_downtime_rftree.pkl', 'rb'))
    model = joblib.load(open('best_downtime_rftree.pkl', 'rb'))
    X_cols = joblib.load(open('downtime_cols.pkl', 'rb'))
    X_pred['building_rating_id'] = X_pred['building_rating_id'].astype('str')

    X = pd.get_dummies(X_pred)
    X = X.reindex(columns=X_cols, fill_value=0)
    scalar = PowerTransformer()
    X = scalar.fit_transform(X)

    y_pred = model.predict(X)
    y_pred = np.round(10** y_pred, 0)
    return y_pred

def renewal_model(X_pred):
    model = joblib.load(open('best_xg_params_pipeline.pkl', 'rb'))
    X_cols = joblib.load(open('renewal_cols.pkl', 'rb'))
    X_pred['building_rating_id'] = X_pred['building_rating_id'].astype('str')
    X_pred['tenant_improvement_allowance_persqft'] = X_pred['tenant_improvement_allowance_persqft']. \
        fillna(0)
    X_pred['free_months'] = X_pred['free_months'].fillna(0)
    X = pd.get_dummies(X_pred)
    X = X.reindex(columns=X_cols, fill_value=0)
    scalar = MinMaxScaler()
    X=scalar.fit_transform(X)

    y_pred = model.predict(X)
    return y_pred

def app():
    st.title('Downtime and Renewal Prediction Comparision')
    data_df, cluster = load_data()

    unique_cbsa_state = sorted(pd.unique((data_df['cbsa_state_new'])))
    select_cbsastate = st.sidebar.selectbox('Select State: ', unique_cbsa_state)

    get_cbsacity = data_df[data_df['cbsa_state_new'] == select_cbsastate]
    unique_cbsa = sorted(pd.unique((get_cbsacity['cbsa_cities'])))
    select_cbsacity = st.sidebar.selectbox('Select cbsa cities: ', unique_cbsa)

    get_cbsaid= data_df[data_df['cbsa_cities'] == select_cbsacity]
    unique_cbsaid = sorted(pd.unique(get_cbsaid['cbsaid']))
    select_cbsaid = st.sidebar.selectbox('Select cbsa id: ', unique_cbsaid)

    get_submarket = get_cbsaid[get_cbsaid['cbsaid'] == select_cbsaid]
    unique_submarkets = sorted(pd.unique(get_submarket['submarket_name']))
    select_submarket = st.sidebar.selectbox("Select submarket", unique_submarkets)

    get_property_id = get_submarket[get_submarket['submarket_name'] == select_submarket]
    unique_property_id = pd.unique(get_property_id['property_id'])
    select_propertyid = st.sidebar.selectbox("Select property id", unique_property_id)

    get_df = get_property_id[get_property_id['property_id'] == select_propertyid]
    get_df = get_df[(get_df['year_off_market'] > 2007) & (get_df['year_off_market'] < 2021)]

    recession ='N'

    renewal = renewal_data()

    renewal = renewal[(renewal['property_id'] == select_propertyid)]
    renewal = renewal[(renewal['new_renewal_from_year'] > 2007) &
                         (renewal['new_renewal_from_year'] < 2021)]
    renewal['age_of_building'] = datetime.datetime.now().year - renewal['construction_year']
    renewal = renewal[renewal['occupied_months'].notnull()]

    submit_btn = st.sidebar.button('Get Predictions')

    downtime_predictors = get_df[['sqft_max', 'estimated_rent',
                                  'building_rating_id', 'Rent_bin', 'month_on_market',
                                  'age_bldg_bins']]
    renewal_predictors = renewal[[ 'estimated_rent', 'rba',
                                  'tenant_improvement_allowance_persqft',
                                  'free_months',
                                  'age_of_building',
                                  'building_rating_id', 'occupied_months',
                                  'Rent_bin']]

    df = get_df[[
                 'sqft_max', 'estimated_rent',
                  'month_on_market', 'vacant_months'
                ]]

    df = df.rename(columns={"vacant_months": "Actual Downtime",
                            "sqft_max": "Sqft",
                            "month_on_market": "Month on Market",
                            "estimated_rent": "Estimated Rent"})

    df_renewal = renewal[['rba', 'estimated_rent', 'occupied_months', 'renewal' ]]
    df_renewal = df_renewal.rename(columns={'renewal': 'Actual Renewal',
                                            'estimated_rent': 'Estimated Rent',
                                            'rba': 'RBA',
                                            'occupied_months': 'Lease Term in months'})

    if submit_btn:
        with st.spinner("Prediction in Progress"):
            st.subheader('Property Characteristics')
            st.text(('Property id: ' +  str(select_propertyid)))
            st.text('Submarket: ' +  select_submarket)
            st.text('City: ' + select_cbsacity)
            st.text('Building rating: ' + str(pd.unique(get_df['building_rating_id'])[0]))
            st.text('Submarket Cluster: ' + pd.unique(get_df['Rent_bin'])[0])
            st.text('Construction year: ' + str(int(pd.unique(get_df['construction_year'])[0])))
            st.header('Downtime Predictions')
            y_pred = downtime_model(downtime_predictors)
            df['Actual Downtime'] = df['Actual Downtime'].astype('int')
            df['Predicted Downtime'] = y_pred.astype('int')
            df['% variation'] = (df['Predicted Downtime'] - df['Actual Downtime'])/ df['Actual Downtime']
            st.dataframe(df.style.format(subset=['Sqft', 'Estimated Rent',
                                                 '% variation'], formatter="{:.2f}"))

            if df_renewal.shape[0] > 0:
                st.header('Renewal Predictions')
                y_renewal = renewal_model(renewal_predictors)
                df_renewal['Predicted Renewal'] = y_renewal
                df_renewal['Predicted Renewal'] = df_renewal['Predicted Renewal'] .apply(lambda x: \
                                                'Y' if x == 1 else 'N')
                df_renewal['Actual Renewal'] = df_renewal['Actual Renewal'].apply(lambda x: \
                                                'Y' if x == 1 else 'N')

                st.dataframe(df_renewal.style.format(subset=['RBA', 'Lease Term in months',
                                          'Estimated Rent'], formatter="{:.2f}"))