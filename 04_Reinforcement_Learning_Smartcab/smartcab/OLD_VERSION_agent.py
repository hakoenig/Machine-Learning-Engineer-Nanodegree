import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

#Defining global variables
possible_actions = [None, 'forward', 'left', 'right']


class LearningAgent(Agent):
    """
    An agent that learns to drive in the smartcab world.
    """

    def __init__(self, env, alpha, epsilon, gamma):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO-DONE: Initialize any additional variables here
        self.q_learner = Q_Learner(alpha, epsilon, gamma, possible_actions)
        self.next_waypoint = self.state = None
        self.collected_reward = self.steps = self.total_neg_reward = 0
        self.trial_count = 0

        # attributes for metrics
        self.steps_taken_list = []
        self.neg_rewards_list = []
        self.neg_rew_per_step = []

        
    def reset(self, destination=None):

        # storing steps previously taken
        self.steps_taken_list.append(self.steps)
        self.neg_rewards_list.append(self.total_neg_reward)
        if self.steps > 0:
            self.neg_rew_per_step.append(self.total_neg_reward / self.steps)
        else:
            self.neg_rew_per_step.append(0)

        self.trial_count += 1

        self.planner.route_to(destination)
        # TODO-DONE: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = self.state = None
        self.collected_reward = self.steps = self.total_neg_reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.steps += 1

        # TODO-DONE: Update state
        # Our learner will chose an action based on the next waypoint and the inputs.
        # The inputs contain traffic information and will only be the values from the inputs dict.
        # In order to be able to retrieve the q value for a given input combination, we will transform
        # the dict to a tuple and store it in self.state 
        self.state = tuple(inputs.items())

        # TODO: Select action according to your policy
        # At first, we will randomly select an action
        #action = random.sample(set(possible_actions), 1)[0]

        # Later, we will ask our q learner, what the best action for given inputs would be.
        # In order to do that, we initialized a q learner class instance as an attribute of the 
        # agent class. The values of the q learner instance should not be resetted after a deadline is done
        # because we don't want to lose the learning progress.
        # In order to choose the next action, our q learner will need to know inputs and next_waypoint.

        # Now we start using our q learner to find the next action
        action = self.q_learner.get_next_action(self.state, self.next_waypoint, self.trial_count)

        # Execute action and get reward
        reward = self.env.act(self, action)

        if reward < 0:
            self.total_neg_reward += reward

        # TODO: Learn policy based on state, action, reward
        # After execution of the step and collection of the reward, we have to let our learner update.
        # We need to pass the new state to our learner, so that it can take the future value of our previous
        # action into account.
        new_state = tuple(self.env.sense(self).items())
        self.q_learner.update_learner(new_state, self.state, self.next_waypoint, action, reward, self.trial_count)

        # For completeness, we will store the sum of collected rewards. However, in this implementation
        # the sum of the earned rewards does not matter for the q learner because we are not taking
        # the sum of earned rewards into account when calculating next actions.
        self.collected_reward += reward
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, nx_way: {}, action = {}, reward = {}, coll_reward: {}".format(deadline, inputs, self.next_waypoint, action, reward, self.collected_reward)  # [debug]


class Q_Learner():
    """
    Class for q learning algorithm with following functions:
        - get_next_action: returns action that agent should take next
        - update_learner: updates the learner with new information about reward for action taken
    """

    def __init__(self, alpha, epsilon, gamma, possible_actions):
        """
        Initializes isntance of q_learner class with alpha, epsilon, gamma and possible actions as arguments.
        
        In:
            - alpha (float): learning rate
            - epsilon (float): probability to choose random action
            - gamma (float): discount rate
            - possible_actions: list of possible actions agent can take
        """
        
        #setting alpha:
        if not isinstance(alpha, (int, float)):
            raise ValueError("Not a valid type. Alpha should be int or float.")
        if alpha > 1 or alpha < 0:
            raise ValueError("Please choose alpha value between 0 and 1.")
        self._alpha = alpha

        #setting epsilon:
        if not isinstance(epsilon, (int, float)):
            raise ValueError("Not a valid type. Epsilon should be int or float.")
        if epsilon > 1 or epsilon < 0:
            raise ValueError("Please choose epsilon value between 0 and 1.")
        self._epsilon = epsilon

        #setting gamma:
        if not isinstance(gamma, (int, float)):
            raise ValueError("Not a valid type. Gamma should be int or float.")
        if gamma > 1 or gamma < 0:
            raise ValueError("Please choose gamma value between 0 and 1.")
        self._gamma = gamma
        
        #setting possible_actions
        self.possible_actions = possible_actions
        
        #intializing dict to store action to values pairs
        #the dict will take a combination of a tuple of states and an action as keys
        self._action_to_value_dict = {}

    #getter for alpha
    @property
    def alpha(self):
        return self._alpha

    #getter for epsilon
    @property
    def epsilon(self):
        return self._epsilon

    #getter for gamma
    @property
    def gamma(self):
        return self._gamma

    #getter for action_to_value_dict
    @property
    def action_to_value_dict(self):
        return self._action_to_value_dict
    
    
    def get_q_value(self, state, next_waypoint, action):
        """
        Returns q value for an action from a state.
        
        In:
            - state (tuple): tuple of inputs for agent
            - next_waypoint (str): direction agent should head next
            - action (str or None): an action the agent can take from current state
        
        Out:
            - q_value of action (float)
        """

        return self.action_to_value_dict.get((state, next_waypoint, action), 0)

    
    def get_next_action(self, state, next_waypoint, trial_count):
        """
        Picks the next action that the agent should take based on learned q values.
        Uses global definition of possible_actions to choose action.

        In:
            - state (tuple): tuple of inputs for agent
            - next_waypoint (str): direction agent should take next
            - trial_count (int): number of trials that have taken place
        
        Out:
            - next_action (str or None)
        """

        # in order to learn rewards for all possible actions, we want to randomly select actions
        # with probability epsilon / trial_count. By diving by trial count we are decreasing the probability
        # of picking randomly smaller and smaller with every trial. 
        # However, this approach does not take into account that we might be exploring a relatively new area of 
        # states that we haven't seen before. In those case, we would want to explore more. Given that our grid is relatively
        # small, we are not going to focus on a solution to this. For future reference though:
        # http://tokic.com/www/tokicm/publikationen/papers/AdaptiveEpsilonGreedyExploration.pdf
        if random.random() < (self.epsilon / trial_count):
            return random.choice(possible_actions)

        q_values_of_actions = [self.get_q_value(state, next_waypoint, action) for action in possible_actions]

        # we could get multiple actions with same q_value. we want to randomly choose between them.
        index_of_action = random.choice([i for i, x in enumerate(q_values_of_actions) if x == max(q_values_of_actions)])

        return possible_actions[index_of_action]
        

    def update_learner(self, new_state, old_state, prev_waypoint, action_taken, reward_received, trial_count):
        """
        Updates the q value for old_state, prev_waypoint, and action_taken combination.

        In:
            - new_state: state after previous move was made
            - old_state: state before previous move was made
            - prev_waypoint: direction that agent was supposed to take
            - action_taken: action that agent took
            - reward_received: reward received for action taken
        """

        previous_q_value = self.action_to_value_dict.get((old_state, prev_waypoint, action_taken), None)

        all_future_vals = []
        for action in possible_actions:
            for potential_waypoint in possible_actions[1:]:
                all_future_vals.append(self.get_q_value(new_state, potential_waypoint, action))

        est_future_val = random.choice([x for x in all_future_vals if x == max(all_future_vals)])

        if previous_q_value:

            # also using a time decay for alpha
            updated_q_value = previous_q_value + (self.alpha / trial_count) * (reward_received + self.gamma * est_future_val - previous_q_value)
            self.action_to_value_dict[(old_state, prev_waypoint, action_taken)] = updated_q_value
        
        else:
            updated_q_value = self.alpha * (reward_received + self.gamma * est_future_val)
            self.action_to_value_dict[(old_state, prev_waypoint, action_taken)] = reward_received


    def __repr__(self):
        """
        Prints information about instance.
        """
        return "Instance of Q_Learner class with alpha: {}, epsilon: {}, gamma: {}, and possible_actions: {}".format(self.alpha, self.epsilon, self.gamma, self.possible_actions)

def run():
    """Run the agent for a finite number of trials."""

    combination_list = []
    final_neg_rew_per_step_list = []
    final_step_amt_list = []

    for alpha in np.arange(0,1.1,0.1):
        for gamma in np.arange(0,1.1,0.1):
            for epsilon in np.arange(0,1.1,0.1):

                print("Testing with {}, {}, {}".format(alpha, gamma, epsilon))

                #run tests

                # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)
                a = e.create_agent(LearningAgent, alpha, epsilon, gamma)  # create agent
                e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
                # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                # Now simulate it
                sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
                # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                sim.run(n_test=100)  # run for a specified number of trials
                # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

                a.steps_taken_list.append(a.steps)
                a.neg_rewards_list.append(a.total_neg_reward)
                if a.steps > 0:
                    a.neg_rew_per_step.append(a.total_neg_reward / a.steps)
                else:
                    a.neg_rew_per_step.append(0)

                combination_list.append((alpha, gamma, epsilon))
                final_neg_rew_per_step_list.append(np.mean(a.neg_rew_per_step[40:]))
                final_step_amt_list.append(np.mean(a.steps_taken_list[40:]))

    best_combo_neg_rew = combination_list[final_neg_rew_per_step_list.index(max(final_neg_rew_per_step_list))]

    print("\n------\nbest combination for neg rew is: {}".format(best_combo_neg_rew))
    print("It achieved a mean value of: {}".format(max(final_neg_rew_per_step_list)))

    best_combo_final_step_amt = combination_list[final_step_amt_list.index(min(final_step_amt_list))]

    print("\n------\nbest combination for least mean step amt is: {}".format(best_combo_final_step_amt))
    print("It achieved a mean value of: {}".format(min(final_step_amt_list)))


if __name__ == '__main__':
    run()
