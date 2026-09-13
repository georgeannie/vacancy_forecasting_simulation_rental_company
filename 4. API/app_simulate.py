import streamlit as st
import pandas as pd
import numpy as np
import simulate_inputs as sm
import datetime
import matplotlib.pyplot as plt
from collections import Counter

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide", page_title="Downtime Simulation")

# LOADING DATA
DATE_TIME = "date/time"

@st.cache(persist=True)
def load_data():
    data_df, cluster = sm.downtime_data()
    return data_df, cluster

@st.cache(persist=True)
def orig_data():
    orig_df = pd.read_csv('lease_clean_Nov22.csv')
    orig_df['age_of_building'] = datetime.datetime.now().year - orig_df['construction_year']

    orig_df['building_rating_id'] = orig_df['building_rating_id'].astype('str')
    orig_df['cbsaid'] = orig_df['cbsaid'].astype('str')
    orig_df['age_bldg_bins'] = orig_df['construction_year'].apply(lambda x: 'New' if x > 2000 else
                    ('Old' if x < 1960 else 'In-Between'))
    orig_df['month_on_market'] = pd.DatetimeIndex(orig_df['date_on_market']).month.astype('str')

    cluster = pd.read_csv('cluster_submarket_Nov22.csv', low_memory=False)
    cluster = cluster[['cbsaid', 'submarket_name', 'Rent_bin', 'Rent']]
    cluster['cbsaid'] = cluster['cbsaid'].astype('str')
    orig_df = pd.merge(orig_df, cluster, left_on=['cbsaid', 'submarket_name'],
                        right_on=['cbsaid', 'submarket_name'], how='left')

    return orig_df

def valuelabel(y,x):
    for i in range(len(y)):
        plt.text(y[i], x[i], y[i])

def prob_rating(df):

    r1 = np.round(df[df['building_rating_id'] == '1'].shape[0] * 1.0 / df.shape[0], 2)
    r2 = np.round(df[df['building_rating_id'] == '2'].shape[0] * 1.0 / df.shape[0], 2)
    r3 = np.round(df[df['building_rating_id'] == '3'].shape[0] * 1.0 / df.shape[0], 2)
    r4 = np.round(df[df['building_rating_id'] == '4'].shape[0] * 1.0 / df.shape[0], 2)
    r5 = 1.0 - (r1 + r2 + r3 + r4)

    return r1, r2, r3, r4, r5

def prob_age(df):
    a1 = np.round(df[df['age_bldg_bins'] == 'New'].shape[0] / df.shape[0], 2)
    a2 = np.round(df[df['age_bldg_bins'] == 'In-betweeb'].shape[0] / df.shape[0], 2)
    a3 = 1 - (a1 + a2)
    return a1, a2, a3


def prob_month(df):
    m1 = np.round(df[df['month_on_market'] == 3].shape[0] / df.shape[0], 2)
    m2 = np.round(df[df['month_on_market'] == 6].shape[0] / df.shape[0], 2)
    m3 = np.round(df[df['month_on_market'] == 9].shape[0] / df.shape[0], 2)
    m4 = 1 - (m1 + m2 + m3)

    return m1, m2, m3, m4

def prob_cluster(df):
    c1 = np.round(df[df['Rent_bin'] == 'High Quality Expensive'].shape[0] / df.shape[0],2)
    c2 = np.round(df[df['Rent_bin'] == 'High Quality Less Expensive'].shape[0] / df.shape[0],2)
    c3 = np.round(df[df['Rent_bin'] == 'Medium Quality Expensive'].shape[0] / df.shape[0],2)
    c4 = np.round(df[df['Rent_bin'] == 'Medium Quality Less Expensive'].shape[0] / df.shape[0],2)
    c5 = np.round(df[df['Rent_bin'] == 'Low Quality Expensive'].shape[0] / df.shape[0],2)
    c6 = 1 - (c1 + c2 + c3 + c4 + c5)
    return c1, c2, c3,c4, c5, c6

def app():
    data_df, cluster = load_data()

    st.title("Downtime Simulation")

    select_sim= st.sidebar.number_input('Select simulations',
                             100, 2000000, 20000, 5000)

    select_rent = st.sidebar.slider('Select rent estimated for the market',
                                    1, 300, (10, 250)
                                    )

    select_sqft = st.sidebar.slider('Select Sqft for the market',
                                    1, 500000, (1000, 100000)
                                    )
    orig_df = orig_data()
    r1, r2, r3, r4, r5 = prob_rating(orig_df)
    a1, a2, a3 = prob_age(orig_df)
    m1, m2, m3, m4 = prob_month(orig_df)
    c1, c2, c3, c4, c5, c6 = prob_cluster(orig_df)


    with st.sidebar.expander('Building Rating Probability'):
        rate1, rate2 = st.columns((6,1))
        with rate1:
            rating_1 = st.number_input('Rating 1: ', 0.0, 1.0, value=r1, step=0.05)
            rating_2 = st.number_input('Rating 2: ', 0.0, 1.0, value=r2, step=0.05)
            rating_3 = st.number_input('Rating 3: ', 0.0, 1.0, value=r3, step=0.05)
            rating_4 = st.number_input('Rating 4: ', 0.0, 1.0, value=r4, step=0.05)
            rating_5 = st.number_input('Rating 5: ', 0.0, 1.0, value=r5, step=0.05)

            bldg_error = ""
            if (rating_1 + rating_2 + rating_3 + rating_4 + rating_5) != 1.0:
                bldg_error = st.error("Set probabilities equal to 1")


        with rate2:
            st.text("")

    if not bldg_error:
        building_rating_probs = [rating_1, rating_2, rating_3, rating_4, rating_5]
        building_rating_df = sm.rating(building_rating_probs, select_sim)

    with st.sidebar.expander('Submarket Cluster Probability'):
        cluster1, cluster2 = st.columns((6,1))
        with cluster1:
            cluster_1 = st.number_input('High Quality Expensive: ', 0.0, 1.0, value=c1, step=0.05)
            cluster_2 = st.number_input('High Quality Less Expensive: ', 0.0, 1.0, value=c2, step=0.05)
            cluster_3 = st.number_input('Medium Quality Expensive: ', 0.0, 1.0, value=c3, step=0.05)
            cluster_4 = st.number_input('Medium Quality Less Expensive: ', 0.0, 1.0, value=c4, step=0.05)
            cluster_5 = st.number_input('Low Quality Expensive: ', 0.0, 1.0, value=c5, step=0.05)
            cluster_6 = st.number_input('Low Quality Less Expensive: ', 0.0, 1.0, value=c6, step=0.05)

        cluster_error = ""
        if (cluster_1 + cluster_2 + cluster_3 + cluster_4 + cluster_5 + cluster_6) != 1.0:
            cluster_error = st.error("Set probabilities equal to 1")

        with cluster2:
            st.text("")

    if not cluster_error:
        cluster_probs = [cluster_1,  cluster_2 , cluster_3, cluster_4, cluster_5,  cluster_6]
        cluster_df = sm.cluster_func(cluster_probs, select_sim)

    with st.sidebar.expander('Age of Building Probability'):
        age1, age2 = st.columns((6,1))
        with age1:
            age_1 = st.number_input('New Buildings (after 2000): ', 0.0, 1.0, value=a1, step=0.01)
            age_2 = st.number_input('Between new and old: ', 0.0, 1.0, value=a2, step=0.01)
            age_3 = st.number_input('Old buildings (Before 1960): ', 0.0, 1.0, value=a3, step=0.01)

            age_error = ""
            if (age_1 + age_2 + age_3) != 1.0:
                age_error = st.error("Set probabilities equal to 1")

        with age2:
            st.text("")

    if not age_error:
        age_probs = [age_1, age_2, age_3 ]
        age_df = sm.age_of_building_func(age_probs, select_sim)

    with st.sidebar.expander('Month property was on market Probability'):
        month1, month2 = st.columns((6, 1))
        with month1:
            month_1 = st.number_input('March: ', 0.0, 1.0, value=m1, step=0.01)
            month_2 = st.number_input('June: ', 0.0, 1.0, value=m2, step=0.01)
            month_3 = st.number_input('September: ', 0.0, 1.0, value=m3, step=0.01)
            month_4 = st.number_input('December: ', 0.0, 1.0, value=m4, step=0.01)

            mo_error = ""
            if (month_1 + month_2 + month_3 + month_4) != 1.0:
                mo_error = st.error("Set probabilities equal to 1")

        with month2:
            st.text("")

    if not mo_error:
        month_probs = [month_1, month_2, month_3,  month_4]
        month_df = sm.month_on_market_func(month_probs, select_sim)

    submit_btn = st.sidebar.button('Simulation!')

    rent_df, mu_log_rent, sd_log_rent = sm.rent(select_rent[0], select_rent[1],
                                                    data_df['estimated_rent'], select_sim)
    sqft_df = sm.sqft(select_sqft[0], select_sqft[1],
                    data_df['sqft_max'], select_sim)

    with st.expander("Input Rent Distributions"):

        st.markdown("<h4 style='text-align: center; color: black;'>Estimated Rent"
                        "</h4>", unsafe_allow_html=True)


        rent1, rent2 = st.columns((3, 3))
        with rent1:
            fig, ax = plt.subplots(figsize=(3, 2))
            plt.hist(data_df['estimated_rent'], bins=20, color="grey")
            plt.title('Log Normal Distribution of Rent from Data', fontsize=8)
            plt.xlabel('Estimated Rent')
            plt.ylabel('Distribution of Rent from Original Data')
            ax.title.set_size(6)
            ax.xaxis.label.set_size(5)
            ax.yaxis.label.set_size(5)
            plt.xticks(fontsize=4)
            plt.yticks(fontsize=4)
            st.pyplot(fig)

        with rent2:
            fig, ax = plt.subplots(figsize=(3, 2))
            plt.hist(rent_df['rent'], bins=20, color='orange')
            plt.title('Log Normal Distribution of Rent (Random)', fontsize=8)
            plt.xlabel('Estimated Rent (Random)')
            plt.ylabel('Distribution of Rent from Randomly selected rent')
            ax.title.set_size(6)
            ax.xaxis.label.set_size(5)
            ax.yaxis.label.set_size(5)
            plt.xticks(fontsize=4)
            plt.yticks(fontsize=4)
            st.pyplot(fig)


    with st.expander("Input Sqft Distributions"):

        st.markdown("<h4 style='text-align: center; color: black;'>Sqft"
                        "</h4>", unsafe_allow_html=True)

        sqftt3, sqft_4 = st.columns((3, 3))
        with sqftt3:
            fig, ax = plt.subplots(figsize=(3, 2))
            plt.hist(data_df['sqft_max'], bins=20, color='grey')
            plt.title('Exponential Distribution of Max Sqft from Data',fontsize=8 )
            plt.xlabel('Sqft')
            plt.ylabel('Distribution of Sqft from Original Data')
            ax.title.set_size(6)
            ax.xaxis.label.set_size(5)
            ax.yaxis.label.set_size(5)
            plt.xticks(fontsize=4)
            plt.yticks(fontsize=4)
            st.pyplot(fig)

        with sqft_4:
            fig, ax = plt.subplots(figsize=(3, 2))
            plt.hist(sqft_df['sqft_max'], bins=20, color='orange')
            plt.title('Exponential Distribution of Max Sqft (Random input of sqft)', fontsize=8)
            plt.xlabel('Sqft (Random)')
            plt.ylabel('Distribution of sqft from randomly selected sqft')
            ax.title.set_size(6)
            ax.xaxis.label.set_size(5)
            ax.yaxis.label.set_size(5)
            plt.xticks(fontsize=4)
            plt.yticks(fontsize=4)
            st.pyplot(fig)

    with st.expander("Input Probability Distributions for Building Rating & Clusters"):
        col1, col2 = st.columns((3, 4))

        with col1:
            if rating_1 + rating_2 + rating_3 + rating_4 + rating_5 != 1:
                st.header('Distribution unavailable')
            else:
                rating_dist = pd.DataFrame({'building rating': [1, 2, 3, 4, 5],
                                            'Rows': [np.round(rating_1 * select_sim,0),
                                                    np.round(rating_2 * select_sim,0),
                                                    np.round(rating_3 * select_sim,0),
                                                    np.round(rating_4 * select_sim,0),
                                                    np.round(rating_5 * select_sim,0)]})
                fig = plt.figure(figsize=(3, 4))
                plt.barh('building rating', 'Rows', data=rating_dist, color = 'grey')
                plt.xlabel("Count of Random Building Rating")
                plt.ylabel("Building rating")
                ax.title.set_size(6)
                ax.xaxis.label.set_size(5)
                ax.yaxis.label.set_size(5)
                plt.xticks(fontsize=4)
                plt.yticks(fontsize=4)
                valuelabel(rating_dist['Rows'], rating_dist['building rating'])

                st.pyplot(fig)

        with col2:
            if cluster_1 + cluster_2 + cluster_3 + cluster_4 + cluster_5 + cluster_6 != 1:
                st.header('Distribution unavailable')
            else:
                cluster_dist = pd.DataFrame({'Cluster': ['High Quality Expensive',
                                                            'High Quality Less Expensive',
                                                            'Medium Quality Expensive',
                                                            'Medium Quality Less Expensive',
                                                            'Low Quality Expensive',
                                                            'Low Quality Less Expensive'],
                                                'Rows': [np.round(cluster_1 * select_sim, 0),
                                                         np.round(cluster_2 * select_sim, 0),
                                                         np.round(cluster_3 * select_sim, 0),
                                                         np.round(cluster_4 * select_sim, 0),
                                                         np.round(cluster_5 * select_sim, 0),
                                                         np.round(cluster_6 * select_sim, 0)]})
                fig = plt.figure(figsize=(3, 5))
                plt.barh('Cluster', 'Rows', data=cluster_dist, color = 'orange')
                plt.xlabel("Count of Random Cluster")
                plt.ylabel("Cluster")
                valuelabel(cluster_dist['Rows'], cluster_dist['Cluster'])
                ax.title.set_size(5)
                ax.xaxis.label.set_size(5)
                ax.yaxis.label.set_size(5)
                plt.xticks(fontsize=5)
                plt.yticks(fontsize=5)
                st.pyplot(fig)


    if submit_btn:
        with st.spinner("Simulation in Progress"):
            y_pred = sm.simulate(select_sim, rent_df, sqft_df, cluster_df, month_df, building_rating_df, age_df)

        print(y_pred)
        sim_plot1, sim_plot2 = st.columns((3,2))
        with sim_plot1:
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.hist(y_pred, weights=np.zeros_like(y_pred) + 100. / y_pred.size,
                    color = "orange")
            plt.title('% Distribution of vacant months for the simulation')
            plt.xlabel("Vacant Months")
            plt.ylabel("Distribution of vacant months")
            ax.title.set_size(5)
            ax.xaxis.label.set_size(4)
            ax.yaxis.label.set_size(4)
            plt.xticks(fontsize=4)
            plt.yticks(fontsize=4)
            st.pyplot(fig)
        with sim_plot2:
            #y_pred =
            count = Counter(y_pred).items()

            percentages = {x: round(y * 100.0/select_sim,1) for x, y in count}
            perc = pd.DataFrame({'Vacant Months': percentages.keys(),
                                 '% Distrubtion of Vacant Months': percentages.values()})

            st.table(perc.sort_values('% Distrubtion of Vacant Months', ascending=False))
