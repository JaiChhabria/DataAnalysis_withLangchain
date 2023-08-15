import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from scipy.stats import ttest_ind, mannwhitneyu  # Import for Hypothesis Testing
from prophet import Prophet  # Import for Forecasting
from dotenv import load_dotenv

# Load environment variables
#load_dotenv()

# Function to query the AI agent
def query_agent(df, query):
    openai_api_key = st.secrect["OPENAI_API_KEY"]
    llm = OpenAI(openai_api_key=openai_api_key)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    return agent.run(query)

# Function to plot a bar chart using Plotly
def plot_bar_chart(df, x_column, y_column):
    fig = px.bar(df, x=x_column, y=y_column, title=f"Bar Chart: {x_column} vs {y_column}")
    st.plotly_chart(fig)

# ... Define other plotting functions similarly ...

def perform_hypothesis_testing(df, column_1, column_2):
    group_1 = df[df[column_1].notnull()][column_1]
    group_2 = df[df[column_2].notnull()][column_2]
    
    if len(group_1) < 30 and len(group_2) < 30:
        t_stat, p_value = ttest_ind(group_1, group_2)
    else:
        _, p_value = mannwhitneyu(group_1, group_2)
    
    return p_value

def perform_forecasting(df, date_column, value_column, forecast_period):
    df = df.rename(columns={date_column: 'ds', value_column: 'y'})
    
    # Convert 'ds' column to datetime format
    df['ds'] = pd.to_datetime(df['ds'])
    
    model = Prophet()
    model.fit(df)
    
    if forecast_period == 'daily':
        periods = min(90, len(df))
    elif forecast_period == 'weekly':
        periods = min(12, len(df) // 7)
    elif forecast_period == 'monthly':
        periods = min(3, len(df) // 30)
    elif forecast_period == 'yearly':
        periods = min(3, len(df) // 365)
    
    # Create a future dataframe with the correct date range
    future = pd.DataFrame(pd.date_range(start=df['ds'].max(), periods=periods + 1, freq='D'), columns=['ds'])
    
    forecast = model.predict(future)
    return forecast
def main():
    st.title("Data Analysis with Streamlit")
    
    # Upload CSV or Excel data
    data = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if data is not None:
        file_extension = data.name.split(".")[-1]
        if file_extension == "csv":
            try:
                df = pd.read_csv(data, encoding="utf-8")
            except UnicodeDecodeError:
                st.error("Error: Unable to decode file with UTF-8 encoding.")
                st.warning("Try specifying a different encoding:")
                encoding = st.text_input("Enter encoding (e.g., latin1):")
                try:
                    df = pd.read_csv(data, encoding=encoding)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
        elif file_extension == "xlsx":
            df = pd.read_excel(data, engine="openpyxl")
        st.success("File uploaded successfully!")
        
        query = st.text_area("Enter your query")
        query_response = ""
        
        if query:
            query_response = query_agent(df, query)
            st.subheader("Query Response:")
            st.write(query_response)
        
        # Set maximum rows displayed in the table
        st.dataframe(df, height=100)  # Adjust the height as needed
    
        # Data Visualization: Chart Options
        st.subheader("Data Visualization: Chart Options")
        chart_types = ["Bar Chart", "Line Chart", "Scatter Plot", "Heatmap"]
        selected_chart = st.selectbox("Select Chart Type", chart_types)
    
        if selected_chart != "Select Chart Type":
            x_column = st.selectbox("Select X-axis column", df.columns)
            y_column = st.selectbox("Select Y-axis column", df.columns)
    
            if x_column != y_column:
                if selected_chart == "Bar Chart":
                    plot_bar_chart(df, x_column, y_column)
                # ... Call other plotting functions similarly ...
            else:
                st.warning("Please select different columns for X-axis and Y-axis.")
        
        # High-Level Statistical Analysis Options
        st.subheader("High-Level Statistical Analysis Options - Work In Progress")
        analysis_options = ["","Hypothesis Testing", "Forecasting", "Logistic Regression", "Linear Regression", "Classification"]
        selected_analysis = st.selectbox("Select Analysis", analysis_options)

        if selected_analysis == "Hypothesis Testing":
            st.subheader("Hypothesis Testing")
            column_1 = st.selectbox("Select Column 1", df.columns)
            column_2 = st.selectbox("Select Column 2", df.columns)
            p_value = perform_hypothesis_testing(df, column_1, column_2)
            st.write(f"P-value of Hypothesis Testing: {p_value:.10f}")

        if selected_analysis == "Forecasting":
            st.subheader("Forecasting")
            date_column = st.selectbox("Select Date Column", df.columns)
            value_column = st.selectbox("Select Value Column", df.columns)
            forecast_period = st.selectbox("Select Forecast Period", ['daily', 'weekly', 'monthly', 'yearly'])
            
            forecast = perform_forecasting(df, date_column, value_column, forecast_period)
            
            st.write("Forecasting Table:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        elif selected_analysis == "Logistic Regression":
            st.subheader("Logistic Regression")
            st.write("Performing Logistic Regression...")
            x_variable = st.selectbox("Select X variable", df.columns)
            y_variable = st.selectbox("Select Y variable (target)", df.columns)
            df[x_variable].fillna(df[x_variable].mean(), inplace=True)  # Impute missing values
            df[y_variable].fillna(df[y_variable].mean(), inplace=True)
            X = df[[x_variable]]
            y = df[y_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            st.write(f"Accuracy of Logistic Regression: {accuracy:.2f}")

        elif selected_analysis == "Linear Regression":
            st.subheader("Linear Regression")
            st.write("Performing Linear Regression...")
            x_variable = st.selectbox("Select X variable", df.columns)
            y_variable = st.selectbox("Select Y variable (target)", df.columns)
            df[x_variable].fillna(df[x_variable].mean(), inplace=True)  # Impute missing values
            df[y_variable].fillna(df[y_variable].mean(), inplace=True)
            X = df[[x_variable]]
            y = df[y_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = LinearRegression()
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            st.write(f"Accuracy of Linear Regression: {accuracy:.2f}")

        elif selected_analysis == "Classification":
            st.subheader("Classification")
            st.write("Performing Classification...")
            x_variable = st.selectbox("Select X variable", df.columns)
            y_variable = st.selectbox("Select Y variable (target)", df.columns)

            # Perform on-hot encoding for categorical columns
            df = pd.get_dummies(df, columns=[x_variable],drop_first=True)
            
            # Convert categorical labels to numerical values
            label_mapping = {label: i for i, label in enumerate(df[y_variable].unique())}
            df[y_variable] = df[y_variable].map(label_mapping)
            
            df[y_variable].fillna(df[y_variable].mean(), inplace=True)
            X = df.drop(y_variable, axis=1)
            y = df[y_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            st.write(f"Accuracy of Classification: {accuracy:.2f}")

    else:
        st.warning("Please upload a CSV or Excel file.")
    
    

if __name__ == "__main__":
    main()
