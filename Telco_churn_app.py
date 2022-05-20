

from cProfile import label
from email.policy import default
import functools
from locale import D_FMT

from pickletools import float8
from sqlite3 import DatabaseError
from statistics import multimode
from turtle import color
#from msilib import datasizemask
#from pathlib import Path

import streamlit as st
#from st_aggrid import AgGrid
#from st_aggrid.shared import JsCode
#from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import plotly as plt
import plotly.express as px
#from joblib import load
import pickle
import numpy as np
import time






#st.markdown('<html><style>.header { width: 1000px; padding: 60px;text-align: center;background: #1abc9c;color: white;font-size: 30px;}</style></html>', unsafe_allow_html=True)


st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; "> Telco Customer Churn Prediction and Analysis</font></H1></div>', unsafe_allow_html=True)

chart = functools.partial(st.plotly_chart, use_container_width=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df['gender'] = df['gender'].map({1: "Female", 0: "Male"}) 
    df['PaymentMethod'] = df['PaymentMethod'].map({0: "Bank Transfer", 1: "Credit Card", 2: "Electronic Ckeck", 3:"Mailed Check"}) 
    df['Contract'] = df['Contract'].map({0: "Month-to-Month", 1: "One Year", 2: "Two years"}) 

    return df


def churn_predictions(df: pd.DataFrame) -> pd.DataFrame:

    return df

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')


def highlight_churn(val):
    
     color = 'green' if val==0 else 'red'
     return f'background-color: {color}'







def main() -> None:
    #st.header("Telco Customer Churn Prediction :phone: :bar_chart:")


    st.subheader("Upload your CSV from Telco")
    uploaded_data = st.file_uploader(
        "Drag and Drop or Click to Upload", type=".csv", accept_multiple_files=False
    )

    if uploaded_data is None:
        st.info("Using example data. Upload a file above to use your own data!")
        uploaded_data = open("/Users/bothainaa/Desktop/Streamlit Churn App/Train_telco.csv", "r")
    else:
        st.success("Uploaded your file!")

    df = pd.read_csv(uploaded_data)
    with st.expander("Raw Dataframe"):
        st.write(df)

    df = clean_data(df)
    with st.expander("Cleaned Data"):
        st.write(df)

    df = churn_predictions(df)
    with st.expander("Predicted Churn Data"):
        st.write(df)
        

    


    col1, col2, col3 = st.columns(3)

    with col2:
        
        st.download_button(
          label="Download data as CSV",
          data=convert_df(df),
          file_name='Predicted_Churn_Customer_Telco.csv',
          mime='text/csv',
     
         )


    st.sidebar.subheader("Filter by Gender")

    gender = ["All", "Female", "Male"]
    
    gender_selections = st.sidebar.radio(
        "Select Gender to filter", gender)




    st.sidebar.subheader("Filter by Payment Method")
    
    payments_all = ["All"]
    payments = list(df.PaymentMethod.unique())
    payment_method = payments_all + payments
    
    payment_selections = st.sidebar.selectbox(
        "Select Payment Method to filter by", payment_method)


    if gender_selections == "All":
        df = df
    else: 
        if gender_selections == "Female":
            df = df.loc[df["gender"] == "Female"]
        else:
            df = df.loc[df["gender"] == "Male"]


    if payment_selections == "All":
        df = df
    else: 
        df = df.loc[df["PaymentMethod"] == payment_selections]







    st.header("Highlights")

    churn_rate = len(df.loc[df["Churn"] == 1])*100/len(df.customerID.unique())
    churns = len(df.loc[df["Churn"] == 1])
    not_churns = len(df.loc[df["Churn"] == 0])


    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total of Unique Customers",
            f"{df.customerID.nunique():.2f}", delta_color="off"
        )

        st.metric(
            "Churn Rate",
            f"%{churn_rate:.2f}", delta_color="off"
            
            
            )



    with col2:
        st.metric(
            "Average Monthly Charges",
            f"${df.MonthlyCharges.mean():.2f}", delta_color="off"
        )

        st.metric(
            "Total Number of Customer Churn",
            f"{churns:.2f}", delta_color="off"
            )


    with col3:
        st.metric(
            "Total of All Charges",
            f"${df.TotalCharges.sum():.2f}", delta_color="off"
        )

        st.metric(
            "Total Number of loyal Customer",
            f"{not_churns:.2f}", delta_color="off"
            )

    st.subheader("Churns by Gender and Contract Type")
    col1, col2 = st.columns(2)


    
    with col1:
        fig = px.histogram(
          df,
          y="Churn",
          x="gender",
          color="gender"
          
          #color_discrete_sequence=px.colors.sequential.Greens,
          
        )
        fig.update_layout(barmode="group", xaxis={"categoryorder": "total descending"})
        chart(fig)

    with col2:

        fig = px.histogram(
          df,
          y="Churn",
          x="Contract",
          color="Contract"
          
        
        )
        fig.update_layout(barmode="group", xaxis={"categoryorder": "total descending"})
        chart(fig)
    
    st.subheader("Churns by Payment Method")
    
    fig = px.histogram(
          df,
          y="Churn",
          x="PaymentMethod",
          color="PaymentMethod"
          
          #color_discrete_sequence=px.colors.sequential.Greens,
          
        )
    fig.update_layout(barmode="group", xaxis={"categoryorder": "total descending"})
    chart(fig)

    
    


    

    st.subheader("% Total Charges by Payment Method")
    fig = px.pie(df, values="TotalCharges", names="PaymentMethod")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    chart(fig)




    st.subheader("Inspect Customer Information!")
    customers_all = ["All"]
    customer_ids = list(df.customerID.unique())
    customers = customers_all + customer_ids
    
    customer_to_inspect = st.selectbox("Select a Customer ", customers)
    if customer_to_inspect == "All":
        customer_data = df[["customerID","gender","MonthlyCharges", "TotalCharges", "PaymentMethod", "Churn"]]
    else:
        customer_data = df[["customerID","gender","MonthlyCharges", "TotalCharges", "PaymentMethod", "Churn"]].loc[df["customerID"] == customer_to_inspect]

    st.table(customer_data.style.applymap(highlight_churn, subset = ["Churn"]))






if __name__ == "__main__":

    main()