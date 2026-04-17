import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model.pkl")

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Airline Dashboard", layout="wide")

# ----------------------------
# PREMIUM CSS
# ----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e0f7fa, #ffffff);
}
.card {
    background: rgba(255, 255, 255, 0.7);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    background: -webkit-linear-gradient(#0077b6, #00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown("<div class='title'>✈ Airline Passenger Satisfaction Dashboard</div>", unsafe_allow_html=True)

# ----------------------------
# LAYOUT
# ----------------------------
col1, col2 = st.columns([2, 1])

# ----------------------------
# INPUT PANEL
# ----------------------------
with col1:
    st.markdown("<div class='card'><b>Passenger Details</b></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 1, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["First-time", "Returning"])

    with c2:
        travel_type = st.selectbox("Type of Travel", ["Business", "Personal"])
        travel_class = st.selectbox("Class", ["Economy", "Economy Plus", "Business"])
        flight_distance = st.number_input("Flight Distance", 0, 5000, 500)

    with c3:
        total_delay = st.number_input("Total Delay", 0, 1000, 10)

# ----------------------------
# RATINGS
# ----------------------------
st.markdown("<div class='card'><b>Service Ratings (1–5)</b></div>", unsafe_allow_html=True)

def rating(label):
    return st.slider(label, 1, 5, 3)

r1, r2, r3 = st.columns(3)

with r1:
    dep_arr = rating("Departure & Arrival Time")
    booking = rating("Online Booking")
    checkin = rating("Check-in Service")
    boarding = rating("Online Boarding")
    gate = rating("Gate Location")

with r2:
    onboard = rating("On-board Service")
    seat = rating("Seat Comfort")
    legroom = rating("Leg Room")
    clean = rating("Cleanliness")
    food = rating("Food & Drink")

with r3:
    inflight = rating("In-flight Service")
    wifi = rating("Wifi Service")
    entertainment = rating("Entertainment")
    baggage = rating("Baggage Handling")

# ----------------------------
# PREDICT BUTTON
# ----------------------------
if st.button("🚀 Predict Satisfaction"):

    input_data = pd.DataFrame([{
        'Age': age,
        'Class': travel_class,
        'Flight Distance': flight_distance,
        'Departure and Arrival Time Convenience': dep_arr,
        'Ease of Online Booking': booking,
        'Check-in Service': checkin,
        'Online Boarding': boarding,
        'Gate Location': gate,
        'On-board Service': onboard,
        'Seat Comfort': seat,
        'Leg Room Service': legroom,
        'Cleanliness': clean,
        'Food and Drink': food,
        'In-flight Service': inflight,
        'In-flight Wifi Service': wifi,
        'In-flight Entertainment': entertainment,
        'Baggage Handling': baggage,
        'Total Delay': total_delay,
        'Gender': gender,
        'Customer Type': customer_type,
        'Type of Travel': travel_type
    }])

    # ----------------------------
    # PREPROCESSING
    # ----------------------------
    oe = OrdinalEncoder(categories=[['Economy', 'Economy Plus', 'Business']])
    input_data['Class'] = oe.fit_transform(input_data[['Class']])

    input_data = pd.get_dummies(
        input_data,
        columns=['Gender', 'Customer Type', 'Type of Travel'],
        drop_first=True
    )

    model_features = [
        'Age', 'Class', 'Flight Distance',
        'Departure and Arrival Time Convenience', 'Ease of Online Booking',
        'Check-in Service', 'Online Boarding', 'Gate Location',
        'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness',
        'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
        'In-flight Entertainment', 'Baggage Handling', 'Total Delay',
        'Gender_Male', 'Customer Type_Returning', 'Type of Travel_Personal'
    ]

    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_features]

    # ----------------------------
    # PREDICTION
    # ----------------------------
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    # ----------------------------
    # OUTPUT PANEL
    # ----------------------------
    with col2:
        st.markdown("<div class='card'><b>Prediction Result</b></div>", unsafe_allow_html=True)

        if pred == 1:
            st.success("✅ SATISFIED")
            confidence = prob[1]
        else:
            st.error("❌ NOT SATISFIED")
            confidence = prob[0]

        st.write(f"Confidence: **{round(confidence*100,2)}%**")

        # ----------------------------
        # Animated Bar Chart
        # ----------------------------
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Not Satisfied", "Satisfied"],
            y=prob,
            text=[f"{round(p*100,2)}%" for p in prob],
            textposition='auto'
        ))
        fig.update_layout(title="Confidence Levels", yaxis=dict(range=[0,1]))
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------
        # Donut Chart
        # ----------------------------
        fig2 = go.Figure(data=[go.Pie(
            labels=["Not Satisfied", "Satisfied"],
            values=prob,
            hole=0.5
        )])
        fig2.update_layout(title="Prediction Distribution")
        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
st.markdown("<div class='card'><b>Feature Importance</b></div>", unsafe_allow_html=True)

try:
    importances = model.feature_importances_
    features = [
        'Age','Class','Flight Distance','Time Convenience','Online Booking',
        'Check-in','Boarding','Gate','On-board','Seat','Leg Room','Cleanliness',
        'Food','Service','Wifi','Entertainment','Baggage','Delay',
        'Gender','Customer Type','Travel Type'
    ]

    fig3 = px.bar(x=importances, y=features, orientation='h',
                  title="Model Feature Importance")
    st.plotly_chart(fig3, use_container_width=True)
except:
    st.info("Feature importance not available")