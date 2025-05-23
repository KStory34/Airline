import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv('data/cleaned_airline_passenger_satisfaction.csv')
df = df.applymap(lambda x: x[0] if isinstance(x, list) else x)
df = df.apply(pd.to_numeric, errors='coerce')
df = df[~df.apply(lambda row: row.astype(str).str.strip().eq("").all(), axis=1)]

page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions!"])

#Home Page
if page == 'Home':
    st.title("Airline Passenger Satisfaction")
    st.subheader("Welcome to the exploration and modeling of passenger satisfaction on airlines ✈️")
    st.write("""
        In this dataset we are looking at how different experiences through the entire booking experience to the departing experience affect the overall satisfaction with the airline satisfaction rate. We will be looking at some key features like Age, Type of travel, Class, Food and drink, and so much more. Before we completely wrap up the deep dive we will also be making our own predictions on satisfaction possibilities.
    """)

#Data Overview Page
if page == 'Data Overview':
    st.title("Data Overview information")
    st.subheader("Key features and extra information.")
    st.write("""
        Age, Class, and Flight Distance are a few main features of this dataset. Mainly working withobject and integer information. This information allows us to understand the satisfction possibilities for almost every persons experience.
        """)
if st.checkbox("Show DataFrame"):
    st.dataframe(df)

if st.checkbox("Show Shape of Data"):
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Exploratory Data Analysis
if page == 'Exploratory Data Analysis':
    st.title("Visualizations and Insights of the Airline Passenger Satisfaction Dataset")
    st.subheader("Select the type of visualization you'd like to explore:")
    
    # Allow user to select visualization type
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplot'], default=[])

    # Extract numerical columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Histogram Visualization
    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Satisfaction 'Neutral or Dissatisfied' = 0, 'Satisfied' = 1"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    # Box Plot Visualization
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x='satisfaction', y=b_selected_col, title=chart_title, color='satisfaction'))

    # Scatterplot Visualization
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='satisfaction', title=chart_title))

if page == "Model Training and Evaluation" :
    st.title("🛠️ Model Training and Evaluation")
    
    st.subheader("Choose a Machine Learning Model")
    model_option = st.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns=['satisfaction'])
    y = df['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    model.fit(X_train_scaled, y_train)  # Train the model

    # Display results
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Purples')
    st.pyplot(fig)

# Make Predictions Page
if page == "Make Predictions!":
    st.title("✈️ Make Predictions")

    st.subheader("Adjust the values below to make predictions on the Airline Passenger Satisfaction dataset:")

    # User inputs for prediction
    gender = st.selectbox("Gender", ['Male','Female'])
    customer_type = st.selectbox("Customer Type", ['Loyal', 'Disloyal'])
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    type_of_travel = st.slider("Type of Travel, business = 0, personal = 1", min_value=0, max_value=1)
    Class = st.selectbox("Class", ['Business', 'Eco', 'Eco Plus'])
    flight_distance = st.slider("Flight Distance (miles)", min_value=50, max_value=5000, value=1000)
    inflight_wifi_service = st.slider("Inflight WiFi Service", min_value=0, max_value=5, value=3)
    departure_arrival_time_convenience = st.slider("Departure/Arrival Time Convenience", min_value=0, max_value=5, value=3)
    ease_of_online_booking = st.slider("Ease of Online Booking", min_value=0, max_value=5, value=3)
    gate_location = st.slider("Gate Location", min_value=0, max_value=5, value=3)
    food_and_drink = st.slider("Food and Drink", min_value=0, max_value=5, value=3)
    online_boarding = st.slider("Online Boarding", min_value=0, max_value=5, value=3)
    seat_comfort = st.slider("Seat Comfort", min_value=0, max_value=5, value=3)
    inflight_entertainment = st.slider("Inflight Entertainment", min_value=0, max_value=5, value=3)
    on_board_service = st.slider("On-board Service", min_value=0, max_value=5, value=3)
    leg_room_service = st.slider("Leg Room Service", min_value=0, max_value=5, value=3)
    baggage_handling = st.slider("Baggage Handling", min_value=0, max_value=5, value=3)
    checkin_service = st.slider("Check-in Service", min_value=0, max_value=5, value=3)
    inflight_service = st.slider("Inflight Service", min_value=0, max_value=5, value=3)
    cleanliness = st.slider("Cleanliness", min_value=0, max_value=5, value=3)
    departure_delay_minutes = st.slider("Departure Delay (minutes)", min_value=0, max_value=1000, value=0)
    arrival_delay_minutes = st.slider("Arrival Delay (minutes)", min_value=0, max_value=1000, value=0)

    gender_mapping = {'Male': 1, 'Female': 0}
    gender = gender_mapping[gender] 
    customer_type_mapping = {'Loyal': 1, 'Disloyal': 0}
    customer_type = customer_type_mapping[customer_type]
    type_of_travel_mapping = {'Business Travel': 1, 'Personal Travel': 0}
    df['Type of Travel'] = df['Type of Travel'].map(type_of_travel_mapping)
    class_mapping = {'Business': 2, 'Eco': 1, 'Eco Plus': 0}
    Class = class_mapping[Class]
    

 # Prepare user input data
    user_input = pd.DataFrame({
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Age': [age],
        'Type of Travel': [type_of_travel],
        'Class': [Class],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [inflight_wifi_service],
        'Departure/Arrival time convenient': [departure_arrival_time_convenience],
        'Ease of Online booking': [ease_of_online_booking],
        'Gate location': [gate_location],
        'Food and drink': [food_and_drink],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [inflight_entertainment],
        'On-board service': [on_board_service],
        'Leg room service': [leg_room_service],
        'Baggage handling': [baggage_handling],
        'Checkin service': [checkin_service],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
        'Departure Delay in Minutes': [departure_delay_minutes],
        'Arrival Delay in Minutes': [arrival_delay_minutes],
    })
    
    st.write("### Your Input Values")
    st.dataframe(user_input)

    df.dropna()
    model = RandomForestClassifier()
    X = df.drop(columns='satisfaction')
    X = X.apply(pd.to_numeric, errors="coerce")
    X.fillna(X.mean(), inplace=True)
    y = df['satisfaction']

    # Scale the data
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy="mean")
    X_filled = imputer.fit_transform(X)  # Fix NaNs
    X_scaled = scaler.fit_transform(X_filled)
    X_scaled = np.nan_to_num(X_scaled)
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)
    ser_input_scaled = np.nan_to_num(user_input_scaled)
    user_input = user_input.apply(pd.to_numeric, errors="coerce")
    user_input.fillna(user_input.mean(), inplace=True)
    
    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction = model.predict(user_input_scaled)[0]

    # Display the result
    st.write(f"The model predicts the passenger is: **{prediction}**")

