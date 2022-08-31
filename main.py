import pandas as pd
import numpy as np
import pickle
import streamlit as st

import warnings 
warnings.filterwarnings('ignore')

@st.cache(allow_output_mutation = True)
def load_model():
    loaded_models ={} # load models
    with open('churn-models.bin', 'rb') as f_in:
        loaded_models['xgb'],loaded_models['Lgb'],loaded_models['Logistic_reg'],loaded_models['bayes'] = pickle.load(f_in)
loaded_models = load_model()

def preprocessing_single(single_dict):
    df = pd.DataFrame(single_dict,index=[0])

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    string_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    #df.churn = (df.churn == 'yes').astype(int)
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df['totalcharges'] = df['totalcharges'].fillna(0)
    return df

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

def predict_(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

def predict_single(trained_models,df_single):
    preds_table_single = pd.DataFrame()
    for model_name in trained_models: 
            #print(f"==========={model_name}==========")
            model = trained_models[model_name]['model_']
            dv = trained_models[model_name]['dv_']
            
            y_pred_single = predict_(df_single, dv, model)
            
            preds_table_single[model_name] = y_pred_single
            #preds_single =  (y_pred_single>= 0.5)*1
    p_df = preds_table_single.copy()
    p_df['blend3'] = 0.4* p_df.Logistic_reg + 0.4*p_df.bayes + 0.1*p_df.xgb + 0.1*p_df.Lgb
    #p_df['blend10'] = 0.3* p_df.Logistic_reg + 0.5*p_df.bayes + 0.1*p_df.xgb + 0.1*p_df.Lgb
    preds_single =  (p_df['blend3']>= 0.5)*1
    
    return     p_df['blend3'].values[0] #, preds_single[0]





def submit():
    st.title('Customer Churn Predictor')
    st.image("churn.jpg")
    st.header('Enter the characteristics of the Customer:')

    tenure = st.slider('Tenure:', 0.0, 72.0, 32.3)
    monthlycharges = st.slider('MonthlyCharges:', 18.25, 118.65, 64.779)
    totalcharges = st.slider('TotalCharges:', min_value=0.0, max_value=8684.8, value=2277.42)
    contract =  st.radio(
     "Contract",
     ('month-to-month', 'two_year', 'one_year'),horizontal=True)
    
    paperlessbilling =  st.radio(
     "Paperless Billing",
     ('yes', 'no'),horizontal=True)

    paymentmethod = st.selectbox(
     'Payment Method:', ['electronic_check', 'mailed_check', 'bank_transfer_(automatic)',
       'credit_card_(automatic)'])
    
    gender = st.radio(
     "Genre",
     ('male', 'female'),horizontal=True)

    seniorcitizen =  st.radio(
     "SeniorCitizen",
     ('Yes', 'No'),horizontal=True)
    seniorcitizen = int(seniorcitizen=='Yes')

    partner =  st.radio(
     "Partner",
     ('no', 'yes'),horizontal=True)

    dependents =  st.radio(
     "Dependents",
     ('no', 'yes'),horizontal=True)

    internetservice =  st.radio(
     "Internet Service",
     ('fiber_optic', 'dsl', 'no'),horizontal=True)

    phoneservice =  st.radio(
     "Phone Service",
     ('yes', 'no'),horizontal=True)

    multiplelines =  st.radio(
     "Multiplelines",
     ('no', 'yes', 'no_phone_service'),horizontal=True)

    onlinesecurity =  st.radio(
     "Online Security",
     ('no', 'yes', 'no_internet_service'),horizontal=True)

    onlinebackup =  st.radio(
     "Online Backup",
     ('no', 'yes', 'no_internet_service'),horizontal=True)
    
    deviceprotection =  st.radio(
     "Device Protection",
     ('no', 'yes', 'no_internet_service'),horizontal=True)

    techsupport =  st.radio(
     "Techsupport",
     ('no', 'yes', 'no_internet_service'),horizontal=True)

    streamingtv =  st.radio(
     "Streaming Tv",
     ('no', 'yes', 'no_internet_service'),horizontal=True)

    streamingmovies =  st.radio(
     "Streaming Movies",
     ('no', 'yes', 'no_internet_service'),horizontal=True)

    customer = {'customerID': '5575-GNVDE',
                        'gender': gender,
                        'SeniorCitizen': seniorcitizen,
                        'Partner': partner,
                        'Dependents': dependents,
                        'tenure': float(tenure),
                        'PhoneService': phoneservice,
                        'MultipleLines': multiplelines,
                        'InternetService': internetservice,
                        'OnlineSecurity': onlinesecurity,
                        'OnlineBackup': onlinebackup,
                        'DeviceProtection': deviceprotection,
                        'TechSupport': techsupport,
                        'StreamingTV': streamingtv,
                        'StreamingMovies': streamingmovies,
                        'Contract': contract,
                        'PaperlessBilling': paperlessbilling,
                        'PaymentMethod': paymentmethod,
                        'MonthlyCharges':float(monthlycharges),
                        'TotalCharges': totalcharges,
                        #'Churn': 'No'
                    }
    clean_customer = preprocessing_single(single_dict=customer)
    prediction = predict_single(trained_models=loaded_models,df_single=clean_customer)
    #prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    
    result = {
        'churn_probability': float(np.round(prediction,3)),
        'churn': bool(churn)
    }

    if st.button('Predict Churn'):
        #price = predict(carat, cut, color, clarity, depth, table, x, y, z)
        st.success(f'Customer Churning : {result["churn"]} ,Churn Probability {result["churn_probability"]}')
    #topk = ['TotalCharges','tenure','Contract','PaymentMethod']
    #return render_template('result.html',customer_passer = customer , best_predictors_passer = topk , result_passer = result)#return jsonify(result)


if __name__ == '__main__':
    submit()
