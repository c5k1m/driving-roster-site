import numpy as np
from copy import deepcopy
from state import State

np.random.seed(42)
class Node:
    def __init__(self, state, parent):
        self.state = state

        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal

        self.parent = parent
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}

        self.action = []

class MCTS:
    def search(self, initial_state, n_iterations=1000, optimize_distance=True):
        self.root = Node(initial_state, None)
        self.optimize_distance = optimize_distance

        for _ in range(n_iterations):
            # Selection phase
            node = self.select(self.root)

            # Rollout phase
            value = self.rollout(node.state)

            # Backpropagation phase
            self.backpropagate(node, value)
        
        try:
            return self.get_best_move(self.root, 0)
        except:
            pass

    def select(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_move(node, 2)
            
            else:
                return self.expand(node)
        
        return node
    
    def expand(self, node):
        actions = node.state.get_valid_actions()

        for action in actions:
            # Create a new_state so that it doesn't mess everything up
            new_state = State(node.state.drivers, 
                                    node.state.passengers, 
                                    node.state.capacity,
                                    state=node.state)
            
            # Perform selected action
            new_state.next_state(action[0], action[1])
            action_id = str(new_state)
            if action_id not in node.children:
                # Create a new node that has the action
                new_node = Node(new_state, node)
                new_node.action = action
                
                # Add the new node to the parent node
                node.children[action_id] = new_node

                if len(actions) == len(node.children):
                    node.is_fully_expanded = True

                return new_node

        print(action_id in node.children.keys())
        print(action_id)
        print(node.children.keys())
        
        print("Big oopsie!!!")

    def rollout(self, state):
        # Create a temporary state to not mess anything up
        temp_state = State(state=state)

        while not temp_state.is_terminal():
            try:
                # Get a random action
                action = np.random.choice(state.get_valid_actions())

                # Perform action
                temp_state.next_state(action[0], action[1])
            
            except:
                empty_seats = temp_state.get_total_allocation() - min(state.total_capacity, state.num_passengers)
                penalty = -empty_seats * temp_state.penalty_scaler
                # if self.optimize_distance:
                #     return temp_state.get_value() + penalty
                # else:
                #     return temp_state.get_standard_dev() + penalty
                return temp_state.get_pareto_objective()+ penalty
        
        # if self.optimize_distance:
        #     return temp_state.get_value()
        # else:
        #     return temp_state.get_standard_dev()
        # Test pareto objective
        return temp_state.get_pareto_objective()

    def backpropagate(self, node, value):
        # update nodes's up to root node
        while node is not None:
            # update node's visits
            node.visit_count += 1
            
            # update node's score
            node.value_sum += value
            
            # set node to parent
            node = node.parent

    def get_best_move(self, node, exploration_constant):
        # Initialize best value as negative infinity since its a maximization algorithm
        best_value = float("-inf")
        best_moves = []

        for child_node in node.children.values():
            # Get value of current action
            action_value = child_node.value_sum / child_node.visit_count + \
                exploration_constant * np.sqrt(np.log(node.visit_count / child_node.visit_count))
            
            if action_value > best_value:
                best_value = action_value
                best_moves = [child_node]
            
            elif action_value == best_value:
                best_moves.append(child_node)
        
        return np.random.choice(best_moves)