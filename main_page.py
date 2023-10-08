import streamlit as st
import pandas as pd
import numpy as np
from state import State
from mcts_no_priors import MCTS
import warnings
import requests
from stqdm import stqdm
import os
warnings.filterwarnings("ignore")

COLLEGE_2_COORD_DICT = { # Convert College or Apartments to Lat, Long
    'Revelle': (32.87446225, -117.24098209937156),
    'Muir': (32.878014199999996, -117.24124330827681),
    'Marshall': (32.8817384, -117.24125091155366),
    'Warren': (32.88276945, -117.2340480725121),
    'ERC': (32.885091200000005, -117.24220110581237),
    'Sixth': (32.880474750000005, -117.24217019235094),
    'Seventh': (32.8881754, -117.2421452),
    'Pepper Canyon': (32.8792306, -117.2317966667491),
    'Rita Atkinson': (32.87257245, -117.23517887907337)
}

# Keep track of who didn't get assigned
unassigned_list = []

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
            
            description_list = {
                "name": "Which column indicates the driver/rider?",
                "phone_num": "Please choose the phone number column",
                "email": "Please choose the email column",
                "live_on_campus": "Please choose the lives on campus column",
                "judge" : "Please choose the judge column",
                "res_address": "Please choose the residential address column",
                "is_driver":"Please choose the driver column",
                "num_seats": "Please choose the capacity/number of seats column",
                "destination": "Please choose the group/destination column"
            }
            
            # Choose which columns contain information about person's name,
            # phone number, address, etc.
            column_choices = df.columns
            column_header_reference_dict = {} # Used to convert what the programmer
                                                           # knows the variables to be versus
                                                           # what the user sets the variables to be
            for column_header, description in description_list.items():
                header_choice = st.selectbox(description,column_choices)
                if header_choice:
                    column_header_reference_dict[column_header] = header_choice

            
            # Get driver names
            name_header = column_header_reference_dict["name"]
            is_driver_header = column_header_reference_dict["is_driver"]
            driver_names = np.array(df[df[is_driver_header] == True][name_header].values)

            # Get passenger names
            passenger_names = np.array(df[df[is_driver_header] == False][name_header].values)

            # Create strict and flex groupings
            st.divider()
            st.header("Special Accommodations")
            strict_dict = add_groups("Riders Grouped With Drivers Strict", driver_names, passenger_names)
            st.divider()
            flex_dict = add_groups("Riders Grouped With Drivers Flex", driver_names, passenger_names)
            
            button_result = st.button("Click to validate addresses (optional)")
            # Validate coordinates
            if button_result:
                df["location"] = None
                with warnings.catch_warnings(): # Doing sketchy stuff
                    st.write("Validating addresses (this may take a few minutes)")
                    live_on_campus_header = column_header_reference_dict["live_on_campus"]
                    df["location"].loc[df[live_on_campus_header].notnull()] = df[df[live_on_campus_header].notnull()][live_on_campus_header]\
                                                                                                .apply(lambda x : COLLEGE_2_COORD_DICT[x])
                    res_address_header = column_header_reference_dict["res_address"]
                    df["location"].loc[df[res_address_header].notnull()] = df[df[res_address_header].notnull()][res_address_header]\
                                                                                                .apply(get_coord)
                    st.write("Validation complete")
                
                # Notify user if there are invalid addresses
                invalid_rows = df[df["location"] == (None, None)]
                if len(invalid_rows) > 0:
                    st.write("The following people have invalid addresses")
                    invalid_addresses = invalid_rows[name_header]
                    for name in invalid_addresses:
                        st.write(f"- {name}")
            
            st.divider()

            # Runs the algorithm
            run_algo_button = st.button("Create roster")
            
            if run_algo_button:
                # Output list
                allocation_list = []

                # Add a fun spinner
                with st.spinner("Generating roster (this may take a few minutes)"):
                    # Rerun location program
                    df["location"] = None
                    with warnings.catch_warnings(): # Doing sketchy stuff
                        live_on_campus_header = column_header_reference_dict["live_on_campus"]
                        df["location"].loc[df[live_on_campus_header].notnull()] = df[df[live_on_campus_header].notnull()][live_on_campus_header]\
                                                                                                    .apply(lambda x : COLLEGE_2_COORD_DICT[x])
                        res_address_header = column_header_reference_dict["res_address"]
                        df["location"].loc[df[res_address_header].notnull()] = df[df[res_address_header].notnull()][res_address_header]\
                                                                                                    .apply(get_coord)
                    
                    # Drop the rows with invalid addresses
                    invalid_rows = df[df["location"] == (None, None)]
                    invalid_idx = invalid_rows.index
                    df.drop(invalid_idx, inplace=True)

                    # For each unique destination/group 
                    destination_header = column_header_reference_dict["destination"]
                    for dest in df[destination_header].unique():
                        group_df = df[df[destination_header] == dest]
                        alloc = run_algorithm(group_df, strict_dict, flex_dict, column_header_reference_dict)
                        allocation_list.append((dest,alloc))
                    
                    # Create output csv file
                    full_df = []
                    name_header = column_header_reference_dict["name"]
                    email_header = column_header_reference_dict["email"]
                    phone_header = column_header_reference_dict["phone_num"]
                    judge_header = column_header_reference_dict["judge"]
                    for dest, allocations in allocation_list:
                        allocations = {k: v for k, v in sorted(allocations.items(), key= lambda ele: len(ele[1]), reverse=True)}
                        df_list = []
                        for group in range(len(allocations)):
                            alloc_df = pd.DataFrame()
                            group_list = [group + 1]
                            driver = list(allocations.keys())[group]
                            name_list = [driver]
                            email_list = [df[df[name_header] == driver][email_header].values[0]]
                            phone_list = [df[df[name_header] == driver][phone_header].values[0]]
                            judge_list  = [df[df[name_header] == driver][judge_header].values[0]]
                            for passenger in allocations[driver]:
                                group_list.append(np.nan)
                                name_list.append(passenger)
                                email_list.append(df[df[name_header] == passenger][email_header].values[0])
                                phone_list.append(df[df[name_header] == passenger][phone_header].values[0])
                                judge_list.append(df[df[name_header] == passenger][judge_header].values[0])
                            alloc_df["Driving Group"] = group_list
                            alloc_df["Name"] = name_list
                            alloc_df["Email"] = email_list
                            alloc_df["Phone Number"] = phone_list
                            alloc_df["Judge"] = judge_list
                            alloc_df["Group"] = dest
                            df_list.append(alloc_df)
                        if len(df_list) > 1:
                            full_df.append(pd.concat(df_list, axis=0))
                        elif len(df_list) == 1:
                            full_df.append(df_list[0])
                    
                    # Process people who had invalid addresses:
                    invalid_df = pd.DataFrame()
                    invalid_df["Name"] = invalid_rows[name_header]
                    invalid_df["Email"] = invalid_rows[email_header]
                    invalid_df["Phone Number"] = invalid_rows[phone_header]
                    invalid_df["Judge"] = invalid_rows[judge_header]
                    invalid_df["Driving Group"] = np.nan
                    invalid_df["Group"] = "Invalid Addresses"
                    full_df.append(invalid_df)

                    # Process Unassigned People
                    invalid_df = pd.DataFrame()
                    unassigned_df = df[df[name_header].isin(unassigned_list)]
                    invalid_df["Name"] = unassigned_df[name_header]
                    invalid_df["Email"] = unassigned_df[email_header]
                    invalid_df["Phone Number"] = unassigned_df[phone_header]
                    invalid_df["Judge"] = unassigned_df[judge_header]
                    invalid_df["Driving Group"] = np.nan
                    invalid_df["Group"] = "Unassigned"
                    full_df.append(invalid_df)

                    # Final Check
                    if len(full_df) > 1:
                        full_df = pd.concat(full_df,axis=0)
                    elif len(full_df) == 1:
                        full_df = full_df[0]
                    else: 
                        print("No CSV Output")

                    csv = convert_df(full_df)

                    st.download_button(
                        "Download Roster",
                        data=csv,
                        file_name="roster.csv",
                        mime="text/csv",
                    )

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def run_algorithm(group_df, strict_dict, flex_dict, reference_dict):
    # Get all reference headers
    driver_header = reference_dict["is_driver"]
    capacity_header = reference_dict["num_seats"]
    name_header = reference_dict["name"]
    location_header = "location" # This one is special since it's a program made column
    
    # Create driver coordinates, passenger coordinates, and capacity arrays
    drivers = np.array(group_df[group_df[driver_header] == True][location_header].values)
    capacity = np.array(group_df[group_df[driver_header] == True][capacity_header].values)
    passengers = np.array(group_df[group_df["is_driver"] == False][location_header].values)

    # Get driver and passenger names
    driver_names = np.array(group_df[group_df[driver_header] == True][name_header].values)
    passenger_names = np.array(group_df[group_df[driver_header] == False][name_header].values)

    # Create and initialize state
    state = State(drivers,
                      passengers,
                      capacity,
                      driver_names=driver_names,
                      passenger_names=passenger_names,
                      strict_dict=strict_dict,
                      flex_dict=flex_dict)
    state.get_init_state()

    # Check if there are enough seats for the passengers
    if (state.total_capacity < state.num_passengers):
        st.subheader("WARNING: NOT ENOUGH SEATS")
    
    mcts = MCTS()
    # Choose best move every time step
    for _ in stqdm(range(state.num_passengers)):
        best_move = mcts.search(state, 400, True)
        if best_move is None:
            break
        action = best_move.action
        
        state.next_state(action[0], action[1])
    
    # Apply post-processing step
    state.post_process()

    # Get unassigned passengers
    for passenger in passenger_names:
        bool_vals = [1 for alloc in state.get_named_allocation().values() if passenger in alloc]
        total = sum(bool_vals)
        if total == 0:
            unassigned_list.append(passenger)

    return state.get_named_allocation()
    

def allocation_2_csv(allocation_list):
    pass

def get_coord(address):
    base_url = f"https://nominatim.openstreetmap.org/search/{address}?format=json&addressdetails=1&limit=1&polygon_svg=1"
    r = requests.get(base_url).json()
    if len(r) == 0:
        return (None, None)
    lat = float(r[0]["lat"])
    long = float(r[0]["lon"])
    return (lat,long)

def add_groups(group_name, driver_list, passenger_list):
    st.subheader(group_name)
    number = st.number_input("Enter the number of " + group_name + " groups", min_value = 0, max_value = len(driver_list))
    output_dict = {}
    for i in range(1, number + 1):
        group_driver = st.selectbox("Select driver for " + group_name + " group " + str(i), driver_list)
        group_passengers = st.multiselect("Select passengers for " + group_name + " group " + str(i), passenger_list)
        output_dict[group_driver] = group_passengers
    
    return output_dict
            
            
if __name__ == "__main__":
    main()