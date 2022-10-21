import streamlit as st
from PIL import Image
import pickle

model = pickle.load(open('model.pkl', 'rb'))

def show_predict_page():
    img1 = Image.open('svu.png')
    img1 = img1.resize((700,160))
    st.image(img1,use_column_width=False)
    st.title("Loan Status Prediction ")
    st.write("Master in Web Sceince ADM S22")
    st.write("Students: Alaa_190363 - Amjad_201024 - Ahmad_183812 - Mohammad_189093")

    ## Account No.
    account_no = st.text_input('Account number')

    ## Full Name
    fn = st.text_input('Full Name')

    ## For Gender
    st.write("Male = 1, Female = 0")
    gen_display = (0,1)
    gen_options = list(range(len(gen_display)))
    gen = st.selectbox("Gender",gen_options, format_func=lambda x: gen_display[x])

    ## For Marital Status
    st.write("Married = 1, Not Married = 0")
    mar_display = (0,1)
    mar_options = list(range(len(mar_display)))
    mar = st.selectbox("Married", mar_options, format_func=lambda x: mar_display[x])

    ## Dependents
    dep_display = (0,1,2,3)
    dep_options = list(range(len(dep_display)))
    dep = st.selectbox("Dependents",  dep_options, format_func=lambda x: dep_display[x])

    ## For Education
    st.write("Graduate = 1, Not Graduate = 0")
    edu_display = (0,1)
    edu_options = list(range(len(edu_display)))
    edu = st.selectbox("Education",edu_options, format_func=lambda x: edu_display[x])

    ## For Self Employed
    st.write("Yes = 1, No = 0")
    emp_display = (0,1)
    emp_options = list(range(len(emp_display)))
    emp = st.selectbox("Self Employed",emp_options, format_func=lambda x: emp_display[x])

    ## Applicant Monthly Income
    mon_income = st.number_input("Applicant's Income($)",value=0)

    ## Co-Applicant Monthly Income
    co_mon_income = st.number_input("Co-Applicant's Income($)",value=0)

    ## Loan Amount
    loan_amt = st.number_input("Loan Amount",value=0)

    ## Loan Amount Term
    dur_display = ['2 Month','6 Month','8 Month','1 Year','16 Month']
    dur_options = range(len(dur_display))
    dur = st.selectbox("Loan Amount Term",dur_options, format_func=lambda x: dur_display[x])

    ## For Credit Score
    cred_display = (0,1)
    cred_options = list(range(len(cred_display)))
    cred = st.selectbox("Credit History",cred_options, format_func=lambda x: cred_display[x])

    if st.button("Predict"):
        duration = 0
        if dur == 0:
            duration = 60
        if dur == 1:
            duration = 180
        if dur == 2:
            duration = 240
        if dur == 3:
            duration = 360
        if dur == 4:
            duration = 480
        features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred]]
        print(features)
        prediction = model.predict(features)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))
        if ans == 0:
            st.error(
                "Hello: " + fn +" || "
                "Account number: "+account_no +' || '
                'Sorry!, you will not get the loan'
            )
        else:
            st.success(
                "Hello: " + fn +" || "
                "Account number: "+account_no +' || '
                'Congratulations!! you will get the loan'
            )
