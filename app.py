import streamlit as st 
import numpy as np 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd 
import pickle 


model=tf.keras.models.load_model('model.keras')

## LOAD ALL THE NECESSARY PICKLE FILES NOW 

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('OHE_country.pkl','rb') as file:
    OHE_country=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


#3streamlit app 

st.title("Customer churn prediction")

geography=st.selectbox('Geography',OHE_country.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tensure=st.slider('Tenure',0,10)
num_of_products=st.slider('Has Credit Card',1,4)
has_cr_card=st.selectbox('Has credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


# input_data=({
#     'CreditScore':[credit_score],
#     'Gender':[label_encoder_gender.transform([gender])[0]],
#     'Age':[age],
#     'Tenure': [tensure] ,
#     'Balance':[balance],
#     'NumOfProducts':[num_of_products],
#     'HasCrCard':[has_cr_card],
#     'IsActiveMember':[is_active_member],
#     'EstimatedSalary':[estimated_salary],
#     'Geography':[geography]
# })

input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tensure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
    'Geography':[geography]
}




geo_encoded=OHE_country.transform([[input_data['Geography'][0]]])
d_df=pd.DataFrame(geo_encoded,columns=OHE_country.get_feature_names_out(['Geography']))
input_df=pd.DataFrame([input_data])
final_df_data=pd.concat([input_df.reset_index(drop=True),d_df.reset_index(drop=True)],axis=1)
final_df_data=final_df_data.drop(columns=['Geography'])
scaled_input=scaler.transform(final_df_data)

prediction=model.predict(scaled_input)

prediction_prob=prediction[0][0]
st.write(f'Churn probability: {prediction_prob:.2f}')

if prediction_prob > 0.5 :
    st.write("The custormer is likely to churn or exit from the bank")
else:
    st.write("the customer will not churn or leave the bank ")
