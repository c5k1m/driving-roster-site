import numpy as np
import networkx as nx
import pandas as pd
from copy import deepcopy

np.random.seed(42)
class State:
    def __init__(self, 
                    drivers=None, 
                    passengers=None,
                    capacity=None, 
                    driver_names=None, 
                    passenger_names=None, 
                    state=None, 
                    max_distance=None,
                    strict_dict=None,
                    flex_dict=None,
                    group_list=None,
                    separate_list=None):

        # Check if state should be copied
        if state is not None:
            self.__dict__ = deepcopy(state.__dict__)

        else:
            # Set basic data
            self.drivers = drivers
            self.passengers = passengers
            self.capacity = capacity
            self.max_distance = max_distance

            # Constraints
            self.strict_dict = {}
            for k,v in strict_dict.items():
                if k in driver_names:
                    self.strict_dict[k] = [p for p in v if p in passenger_names]
            self.flex_dict = {}
            for k,v in flex_dict.items():
                if k in driver_names:
                    self.flex_dict[k] = [p for p in v if p in passenger_names]
            self.group_list = group_list
            self.separate_list = separate_list

            # Setup calculations
            self.num_drivers = len(drivers)
            self.num_passengers = len(passengers)
            self.num_flex_drivers = len(self.flex_dict)
            self.total = self.num_drivers + self.num_flex_drivers + self.num_passengers
            self.network = np.inf * np.ones((self.total, self.total)) # Keeps track of distances and which nodes are connected to another
            self.lookup = np.zeros((self.num_drivers + self.num_flex_drivers, self.num_passengers)) # Sees who is connected to who but does not encode directionality
            self.total_capacity = self.capacity.sum()

            # Allocation dictionary
            self.allocations = {k:[] for k in range(self.num_drivers + self.num_flex_drivers)} # See who is connected to who with directionality

            # Conversion from id to name
            self.driver2name = {k:name for k, name in enumerate(driver_names)}
            self.passenger2name = {k:name for k, name in enumerate(passenger_names)}

            # Conversion from name to id
            self.name2driver = dict([(v, k) for k,v in self.driver2name.items()])
            self.name2passenger = dict([(v, k) for k,v in self.passenger2name.items()])

            # Conversion from real flex drivers to fake flex drivers
            self.real2fake_flex = {self.name2driver[name]:k + self.num_drivers for k, name in enumerate(self.flex_dict.keys())}
            self.fake2real_flex = dict([(v, k) for k,v in self.real2fake_flex.items()])
    
    def haversine(self,point1, point2): # Converts latitude and longitude into miles
        lat1 = np.deg2rad(point1[0]) # Convert to radians
        lat2 = np.deg2rad(point2[0])
        lon1 = np.deg2rad(point1[1])
        lon2 = np.deg2rad(point2[1])
        dlon = (lon2 - lon1)
        dlat = (lat2 - lat1)
        a = (np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2)
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return 3959*c # Multiply arc length by radius of earth in miles
    
    def get_init_state(self):
        # Assign distances from driver to passengers
        # Drivers can't driver other drivers so their distance will be zero
        for i in range(self.num_drivers):
            for j in range(self.num_passengers):
                passenger = self.num_drivers + self.num_flex_drivers + j
                dist = self.haversine(self.drivers[i], self.passengers[j]) # Haversine distance returns miles
                self.network[i][passenger] = dist
                self.network[passenger][i] = dist
        
        # Assign distances from passenger to passenger
        for i in range(self.num_passengers):
            for j in range(self.num_passengers):
                if i != j:
                    passenger1 = self.num_drivers + self.num_flex_drivers + i
                    passenger2 = self.num_drivers + self.num_flex_drivers + j
                    dist = self.haversine(self.passengers[i],  self.passengers[j])
                    self.network[passenger1][passenger2] = dist
                    self.network[passenger2][passenger1] = dist

        # Assign distances from fake flex drivers to passenger
        for fake in self.fake2real_flex.keys():
            real = self.fake2real_flex[fake] # Get the real driver associated with the fake flex driver
            assigned_passengers = self.flex_dict[self.driver2name[real]] # Get the assigned passenger from real driver
            passenger_coord = [self.passengers[self.name2passenger[passenger]] for passenger in assigned_passengers] # Get Passenger coordinates
            coord_arr = np.array([self.drivers[real], *passenger_coord]) # Create new array using driver and passenger coordinates
            avg_coord = coord_arr.mean(axis=0) # Get average of driver and passenger coordinates

            # Calculate distance from fake driver to passengers
            # Fake driver's position is the average coordinate
            for i in range(self.num_passengers):
                passenger = self.num_drivers + self.num_flex_drivers + i
                dist = self.haversine(avg_coord, self.passengers[i])
                self.network[fake][passenger] = dist
                self.network[passenger][fake] = dist

        self.penalty_scaler = self.network[self.network != np.inf].max()

        # Remove connections for strict groupings
        for driver, passengers in self.strict_dict.items():
            # Get the network ids of both the driver and the passengers
            driver_id = [self.name2driver[driver]]
            strict_idx = [driver_id]
            strict_idx.extend([self.name2passenger[passenger] + self.num_drivers + self.num_flex_drivers for passenger in passengers])
            mask_array = np.zeros_like(self.network[driver_id])
            mask_array[strict_idx] = 1

            # A very convoluted way of removing connections without recalculating distances
            for id in strict_idx:
                self.network[id] = np.where(mask_array == 0, np.inf, self.network[id])
                self.network.T[id] = np.where(mask_array == 0, np.inf, self.network.T[id])

        # Remove connections for real flex groupings
        for driver, passengers in self.flex_dict.items():
            # Get the network ids of both the driver and the passengers
            driver_id = self.name2driver[driver]
            flex_idx = [driver_id]
            flex_idx.extend([self.name2passenger[passenger] + self.num_drivers + self.num_flex_drivers for passenger in passengers])
            mask_array = np.zeros_like(self.network[driver_id])
            mask_array[flex_idx] = 1

            # A very convoluted way of removing connections without recalculating distances
            for id in flex_idx:
                self.network[id] = np.where(mask_array == 0, np.inf, self.network[id])
                self.network.T[id] = np.where(mask_array == 0, np.inf, self.network.T[id])

        # Remove connections for fake flex groupings
        for driver, passengers in self.flex_dict.items():
            # Get the network ids of both the driver and the passengers
            driver_id = self.real2fake_flex[self.name2driver[driver]]
            flex_idx = [driver_id]
            flex_idx.extend([self.name2passenger[passenger] + self.num_drivers + self.num_flex_drivers for passenger in passengers])
            mask_array = np.zeros_like(self.network[driver_id])
            mask_array[flex_idx] = 1

            # A very convoluted way of removing connections without recalculating distances
            self.network[driver_id] = np.where(mask_array == 1, np.inf, self.network[driver_id])
            self.network.T[driver_id] = np.where(mask_array == 1, np.inf, self.network.T[driver_id])
        
        # Update capacity for fake drivers
        for real in self.real2fake_flex.keys():
            real_cap = self.capacity[real] # Real capacity of the driver
            num_assigned = len(self.flex_dict[self.driver2name[real]]) # Number of people assigned to driver
            self.capacity = np.append(self.capacity, real_cap - num_assigned) # Append updated capacity to fake driver
            self.capacity[real] = num_assigned # Reassign real driver's capacity to number of people assigned
        
    def __str__(self):
        out = ""
        for driver, passengers in self.allocations.items():
            driver_txt = self.driver2name[driver] if driver in self.driver2name else driver
            out += f"{driver_txt}: "
            for passenger in passengers:
                out += f"{self.passenger2name[passenger]} -> "
            out += "\n"
        return out
 
    def next_state(self, driver, passenger):
        # I am assuming that all inputs will be valid allocations
        # I am lazy and don't want to put in checks

        self.allocations[driver].append(passenger) # Update allocations
        self.lookup[driver][passenger] = 1 # Update lookup table

    def get_valid_actions(self): 
        out = [] # List of valid actions
        claimed = [] # List of claimed passengers
        
        # Check if a driver already has a passenger allocated to them
        for passengers in self.allocations.values():
            for passenger in passengers:
                claimed.append(passenger)

        for driver in range(self.num_drivers + self.num_flex_drivers):
            # Look up total distance driver has travelled if max_distance is specified
            if self.max_distance is not None:
                total_dist = 0
                passengers = self.allocations[driver] # Get list of every passenger allocated to driver
                last = driver # Keep track of last person picked up in the list
                for passenger in passengers:
                    # We add the distance from the last pickup to the current pickup
                    curr = passenger + self.num_drivers + self.num_flex_drivers
                    total_dist += self.network[last][curr]
                    last = curr

                if total_dist > self.max_distance:
                    continue
            
            # If the driver hasn't reached full capacity then search for who can be picked up
            if self.lookup[driver].sum() < self.capacity[driver]:
                # If driver already picked someone up then see the connections of the last passenger
                if len(self.allocations[driver]) > 0:
                    # Get last passenger picked up
                    last_passenger = self.allocations[driver][-1]  + self.num_drivers + self.num_flex_drivers
                    # We subtract num_drivers since drivers are numbered 0 to n-1 and passengers n to n+m-1
                    # Where n is the number of drivers and m is the number of passengers
                    connections = np.argwhere(self.network[last_passenger] != np.inf) - self.num_drivers - self.num_flex_drivers
                    for connection in connections:
                        # If the connection is greater than or equal to zero then we know its another passenger
                        # Also check if passenger has already been claimed
                        if connection[0] >= 0 and connection[0] not in claimed:
                            # Connection index at zero cuz numpy is dumb
                            out.append((driver, connection[0]))
                
                else: # If the driver hasn't pioked someone up yet then get all of their connections
                    for passenger in np.argwhere(self.network[driver] != np.inf):
                        # We subtract from number of drivers to get the real passenger index
                        real_passenger_idx = passenger[0] - self.num_drivers - self.num_flex_drivers

                        # If the passenger hasn't been claimed yet then add action
                        if real_passenger_idx not in claimed:
                            out.append((driver, real_passenger_idx))
                            
        # Return a list of possible (driver, passenger)
        return out
    
    def _total_distance(self, combined_list):
        # Set total to distance between driver and first passenger because
        # driver is always position 0
        total = self.haversine(self.drivers[combined_list[0]], 
                                     self.passengers[combined_list[1]])
        
        for i in range(1, len(combined_list) - 1):
            p1 = self.passengers[combined_list[i]]
            p2 = self.passengers[combined_list[i+1]]
            total += self.haversine(p1, p2)
        return total

    def _swap(self, combined_list, p1, p2):
        combined_list = deepcopy(combined_list) # PYTHON OBJECTS ARE ALWAYS MUTABLE
        temp = deepcopy(combined_list[p1]) # AAAHHHHHHHHH
        combined_list[p1] = combined_list[p2]
        combined_list[p2] = temp
        return combined_list
    
    def post_process(self):
        # Compress the fake allocations to the real allocations
        for real, fake in self.real2fake_flex.items():
            if(len(self.allocations[fake]) > 0):
                # Re-calculate distance between passengers 
                # because information was lost during separation
                id_1 = self.allocations[real][-1]
                id_2 = self.allocations[fake][0]
                dist = self.haversine(self.passengers[id_1], self.passengers[id_2])
                offset = self.num_flex_drivers + self.num_drivers
                self.network[id_1 + offset][id_2 + offset] = dist
                self.network[id_2 + offset][id_1 + offset] = dist

            self.allocations[real].extend(self.allocations[fake]) # Append the fake results to the real driver
            self.allocations.pop(fake, None) # Pop the fake results

        # Two Opt Swap Optimization
        for driver, passengers in self.allocations.items():
            combined_list = np.array([driver, *passengers])
            best_distance = 0

            # NOTE: Position 0 is always driver
            if len(combined_list) > 2: # Only works for lists of size 3 and above
                best_distance = self._total_distance(combined_list)
                # Start slow pointer at 1 because we don't want to change position 0
                for slow in range(1, len(combined_list) - 2):
                    for fast in range(slow+1, len(combined_list)):
                        new_list = self._swap(combined_list, slow, fast)
                        new_distance = self._total_distance(new_list)
                        if new_distance < best_distance:
                            combined_list = new_list
                            best_distance = new_distance
            
            self.allocations[combined_list[0]] = combined_list[1:]

    def is_terminal(self):
        # We assume all drivers allocated until proven otherwise
        driver_full = True
        for driver in range(self.num_drivers + self.num_flex_drivers):
            if self.lookup[driver].sum() < self.capacity[driver]:
                driver_full = False
                break # Break early since we no longer need to check

        # We assume all passengers are allocated until proven otherwise
        all_allocated = True
        for passenger in range(self.num_passengers):
            if np.transpose(self.lookup)[passenger].sum() == 0:
                all_allocated = False
                break # Break early since we no longer need to check
        
        # No actions?
        no_actions = (len(self.get_valid_actions()) == 0)

        # If all drivers are allocated or all passengers are allocated return true
        # The state is also terminal if the set of actions is zero
        return driver_full or all_allocated or no_actions

    def get_value(self):
        total_dist = 0 # Return total distance driven by all drivers

        for driver, passengers in self.allocations.items():
            last = driver # Keep track of last person picked up in the list
            for passenger in passengers:
                # We add the distance from the last pickup to the current pickup
                curr = passenger + self.num_drivers + self.num_flex_drivers

                total_dist += self.network[last][curr]
                last = curr
        
        # Return negative total distance since it's a maximization algorithm
        return -total_dist

    def get_standard_dev(self):
        dist = [] # List of distances of each driver

        for driver, passengers in self.allocations.items():
            passengers = self.allocations[driver] # Get list of every passenger allocated to driver
            last = driver # Keep track of last person picked up in the list
            driver_dist = 0
            for passenger in passengers:
                # We add the distance from the last pickup to the current pickup
                curr = passenger + self.num_drivers + self.num_flex_drivers
                driver_dist += self.network[last][curr]
                last = curr

            # We don't count drivers who don't have passengers
            if(len(passengers) > 0):
                dist.append(driver_dist)
        
        # Return negative standard deviation since it's a maximization algorithm
        return -np.std(dist)
    
    def get_pareto_objective(self):
        # Return the negative pareto objective since it's a maximization algorithm
        return -np.sqrt(self.get_value()**2 + self.get_standard_dev()**2)
    
    def get_total_allocation(self):
        return int(np.einsum("ij->i",self.lookup).sum())
    
    def get_named_allocation(self):
        out = {}
        for i in range(len(self.allocations)):
            passengers = [self.passenger2name[passenger] for passenger in self.allocations[i]]
            out[self.driver2name[i]] = passengers
        return out

    def to_csv(self, name):
        # Sort groups by largest to smallest
        self.allocations = {k: v for k, v in sorted(self.allocations.items(), key= lambda ele: len(ele[1]), reverse=True)}
        df_list = []
        for group in range(len(self.allocations)):
            df = pd.DataFrame()
            group_list = [group + 1]
            key = list(self.allocations.keys())[group]
            name_list = [f"Driver {key}"]
            email_list = [f"Driver {key}@ucsd.edu"]
            phone_list = [f"({np.random.randint(100, 999)}) {np.random.randint(100, 999)}-{np.random.randint(100, 999)}"]
            for passenger in self.allocations[key]:
                group_list.append(np.nan)
                name_list.append(f"Passenger {passenger}")
                email_list.append(f"Passenger {passenger}@ucsd.edu")
                phone_list.append(f"({np.random.randint(100, 999)}) {np.random.randint(100, 999)}-{np.random.randint(100, 999)}")
            df["Group"] = group_list
            df["Name"] = name_list
            df["Email"] = email_list
            df["Phone Number"] = phone_list
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        df.to_csv(f"{name}.csv", index=False)