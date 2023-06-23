import streamlit as st
import pandas as pd
from state import State
from mcts_no_priors import MCTS
import requests
import asyncio

def main():
    st.title('Driving Roster Automation')
    menu = ["Home", "About Us"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "About Us":
        st.subheader("About Us")
        # mini biographies here
    
    if choice == "Home":
        input_file = st.file_uploader("Upload your CSV file here", type=["csv"])
        if input_file is not None:
            df = pd.read_csv(input_file)
            
            drivers_df = df[(df['is_driver'] == True)]	
            drivers = drivers_df['name'].tolist() # list of drivers
            passengers_df = df[(df['is_driver'] == False)]	
            passengers = passengers_df['name'].tolist() # list of passengers
            
            column_menu = ["Name", "Email", "Is a driver (boolean column)?", "Capacity of car", 
                            "Starting point address", "Destination", "Others"]
                            # phone number? Juge?
            
            st.header("Assign Columns")
            for i in df.columns:
                col_content = st.selectbox("Please identify the content of \"" + i + "\" column", column_menu)
                # store the selection

            st.divider()
            st.header("Special Accommodations")
            add_groups("Riders Grouped With Drivers Strict", drivers, passengers)
            st.divider()
            add_groups("Riders Grouped With Drivers Flex", drivers, passengers)


def add_groups(group_name, driver_list, passenger_list):
    st.subheader(group_name)
    number = st.number_input("Enter the number of " + group_name + " groups", min_value = 0, max_value = len(driver_list))
    for i in range(1, number + 1):
        group_driver = st.selectbox("Select driver for " + group_name + " group " + str(i), driver_list)
        group_passengers = st.multiselect("Select passengers for " + group_name + " group " + str(i), passenger_list)
        # Store all groups
    # return
    


main()