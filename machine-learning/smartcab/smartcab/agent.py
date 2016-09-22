import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import time

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, epsilon):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.Q = {}
        self.epsilon = epsilon
        self.wrongActions = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # map inputs and next_waypoint to a state
        self.state = self.getState(inputs['light'], self.next_waypoint, inputs['oncoming'], inputs['right'], inputs['left'])

        # get the action which maximizes Q(s, action)
        action = max(['left', 'right', 'forward', None], key=lambda a: self.getQ(self.state, a))

        if random.random() < self.epsilon:
            action = random.choice(['left', 'right', 'forward', None])

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            self.wrongActions += 1

        # TODO: Learn policy based on state, action, reward
        nextInputs = self.env.sense(self)
        nextWaypoint = self.planner.next_waypoint()
        next_state = self.getState(nextInputs['light'], nextWaypoint, nextInputs['oncoming'], nextInputs['right'], nextInputs['left'])
        maxFutureQ = max(map(lambda a: self.getQ(next_state, a), ['left', 'right', 'forward', None]))

        # if nexts == None then that means we don't have Q values for the next states, so don't take it into account
        if maxFutureQ == None:
            maxFutureQ = 0

        gamma = 0.01
        alpha = 0.2

        currentQ = self.Q[self.state][action]
        if currentQ == None:
            currentQ = 0

        newQ = reward + gamma * maxFutureQ
        self.Q[self.state][action] = (1-alpha) * currentQ + alpha * newQ

        print("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward))  # [debug]

    def getState(self, light, direction, oncoming, right, left):
        trafficCrosses =  \
            (direction == 'forward' and right    in ['forward', 'right', 'left']) or \
            (direction == 'forward' and oncoming in ['right']) or \
            (direction == 'forward' and left     in ['forward', 'left']) or \
            (direction == 'left'    and right    in ['forward', 'left']) or \
            (direction == 'left'    and oncoming in ['forward', 'right']) or \
            (direction == 'left'    and left     in ['forward', 'left']) or \
            (direction == 'right'   and oncoming in ['right']) or \
            (direction == 'right'   and left     in ['forward'])

        return "{}-{}-{}".format(trafficCrosses, light, direction)

    def getQ(self, state, action):
        if not state in self.Q:
            self.Q[state] = {}
        if not action in self.Q[state]:
            self.Q[state][action] = None

        return self.Q[state][action]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(10)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, 0.8)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "Wrong actions: {}".format(a.wrongActions)
    a.wrongActions = 0
    a.epsilon = 0.0
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    sim.run(n_trials=100)  # run for a specified number of trials
    print "Wrong actions: {}".format(a.wrongActions)


if __name__ == '__main__':
    run()
