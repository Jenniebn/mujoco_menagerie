import mujoco
import numpy as np
from mujoco.viewer import launch_passive

model_path = "mjx_single_cube.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Actuator indices based on your XML definition:
#   actuator1 -> joint1
#   actuator2 -> joint2
#   actuator3 -> joint3
#   actuator4 -> joint4
#   actuator5 -> joint5
#   actuator6 -> joint6
#   actuator7 -> joint7 (arm joints)
#   actuator8 -> finger_joint1 (gripper width), finger_joint2 tied by equality
arm_actuators = range(7)
gripper_actuator = 7  # The 8th actuator (index=7) controls the gripper

def get_keyframe_robot_joints(model, key_name):
    """Extract robot and gripper joint positions from a named keyframe, ignoring the object state."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    key_qpos = model.key_qpos[key_id, :]
    # key_qpos format (assuming):
    # [7 arm joints, 2 finger joints, 3 for box pos, 4 for box quat] = total 16
    # We only care about the first 9 (7 arm + 2 finger)
    robot_joints = key_qpos[:7]
    gripper_joints = key_qpos[7:9]  
    return robot_joints, gripper_joints

def move_robot(model, data, target_arm_positions, steps=5000):
    """Move the robot arm to the target positions using position actuators over a number of steps."""
    start_positions = np.copy(data.qpos[:7])
    for i in range(steps):
        alpha = i / (steps - 1)
        desired_positions = (1 - alpha)*start_positions + alpha*target_arm_positions
        # Set arm actuators
        data.ctrl[:7] = desired_positions  
        # Step the simulation
        mujoco.mj_step(model, data)
        yield

def set_gripper(model, data, target_width, steps=2000):
    """Close or open the gripper by setting its actuator ctrl value from current to target_width."""
    start_width = data.ctrl[gripper_actuator]
    for i in range(steps):
        alpha = i / (steps - 1)
        data.ctrl[gripper_actuator] = (1 - alpha)*start_width + alpha*target_width
        mujoco.mj_step(model, data)
        yield

def visualize_simulation():
    viewer = launch_passive(model, data)
    steps_per_transition = 5000
    gripper_steps = 2000

    # Move the arm to the 'home' position
    home_arm, home_gripper = get_keyframe_robot_joints(model, "home")
    print("Moving arm to home position...")
    for _ in move_robot(model, data, home_arm, steps_per_transition):
        viewer.sync()

    # Open the gripper fully (assume 0.04 is fully open)
    data.ctrl[gripper_actuator] = 0.04
    for _ in range(1000):
        mujoco.mj_step(model, data)
        viewer.sync()

    # Move the robot close to the box (pre-grasp position)
    pickup1_arm, pickup1_gripper = get_keyframe_robot_joints(model, "pickup")
    print("Moving to pickup1 (pre-grasp) position...")
    for _ in move_robot(model, data, pickup1_arm, steps_per_transition):
        viewer.sync()

    # Close the gripper around the box (0.0 is fully closed)
    print("Closing the gripper...")
    for _ in set_gripper(model, data, 0.0, gripper_steps):
        viewer.sync()

    # Let the simulation settle to ensure the box is grasped
    print("Holding gripper closed to ensure grasp...")
    for _ in range(200):
        mujoco.mj_step(model, data)
        viewer.sync()

    # Lift the box by moving to the 'pickup' keyframe arm position
    # pickup_arm, pickup_gripper = get_keyframe_robot_joints(model, "pickup")
    # print("Lifting the box...")
    # for _ in move_robot(model, data, pickup_arm, steps_per_transition):
    #     viewer.sync()

    # Return to home position with the box
    print("Returning to home position...")
    for _ in move_robot(model, data, home_arm, steps_per_transition):
        viewer.sync()

    print("Simulation completed. Close the window to exit.")
    while True:
        viewer.sync()

# Run the visualization
visualize_simulation()
