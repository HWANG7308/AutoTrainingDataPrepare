import ompl.base as ob
import ompl.geometric as og
import numpy as np
import math3d as m3d
import math

from PoseGenerator import PoseGenerator
from URController import UR5RobotController

# Define the UR5 joint limits
joint_limits = [
    (-2 * np.pi, 2 * np.pi),  # Joint 1
    (-2 * np.pi, 2 * np.pi),  # Joint 2
    (-2 * np.pi, 2 * np.pi),  # Joint 3
    (-2 * np.pi, 2 * np.pi),  # Joint 4
    (-2 * np.pi, 2 * np.pi),  # Joint 5
    (-2 * np.pi, 2 * np.pi),  # Joint 6
]

# Define the state space for the UR5 robot
space = ob.RealVectorStateSpace(6)
bounds = ob.RealVectorBounds(6)
for i, (low, high) in enumerate(joint_limits):
    bounds.setLow(i, low)
    bounds.setHigh(i, high)
space.setBounds(bounds)


# Define a simple validity checker
def is_state_valid(state):
    # Here you can add checks for collisions or other constraints
    return True


# Create a space information object
si = ob.SpaceInformation(space)
si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))

# Define the start and goal states
start = ob.State(space)
goal = ob.State(space)

# Example poses (replace with your actual poses)
poses = [
    [0, -np.pi / 4, np.pi / 2, -np.pi / 4, np.pi / 2, 0],
    [np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 4, np.pi / 2, np.pi / 4],
    # Add more poses as needed
]


# Create a UR5 robot controller
ROBOT_IP = "192.168.2.144"  # URSim
UR5 = UR5RobotController(ROBOT_IP)

T_rob2obj = m3d.Transform(
    m3d.Orientation.new_rotation_vector((math.pi / 2, 0, 0)), m3d.Vector(0, -0.6, 0)
)
T_end2cam = m3d.Transform(
    m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
)

# Generate the end-effector positions to capture object images from various defined views
robot_poses = PoseGenerator(T_rob2obj, T_end2cam).generate_positions()

# Set the start and goal states
start_values = poses[0]
goal_values = poses[-1]
for i in range(6):
    start[i] = start_values[i]
    goal[i] = goal_values[i]

# Create the problem definition
pdef = ob.ProblemDefinition(si)
pdef.setStartAndGoalStates(start, goal)

# Create the RRT planner
planner = og.RRT(si)
planner.setProblemDefinition(pdef)
planner.setup()

# Solve the problem
solved = planner.solve(1.0)

if solved:
    # Get the planned path
    path = pdef.getSolutionPath()
    print("Found solution:")
    path.printAsMatrix()
else:
    print("No solution found")

# Optionally, you can interpolate the path for smoother motion
if solved:
    path.interpolate()
    print("Interpolated path:")
    path.printAsMatrix()
