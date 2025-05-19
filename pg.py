import os
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


def choose_mimicgen_environment():
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """

    # try to import robosuite task zoo to include those envs in the robosuite registry
    try:
        import robosuite_task_zoo
    except ImportError:
        pass

    # all base robosuite environments (and maybe robosuite task zoo)
    robosuite_envs = set(suite.ALL_ENVIRONMENTS)

    # all environments including mimicgen environments

    import mimicgen
    all_envs = set(suite.ALL_ENVIRONMENTS)

    # get only mimicgen envs
    only_mimicgen = sorted(all_envs - robosuite_envs)

    # keep only envs that correspond to the different reset distributions from the paper
    envs = [x for x in only_mimicgen if x[-1].isnumeric()]

    # Select environment to run
    print("Here is a list of environments in the suite:\n")

    # for k, env in enumerate(envs):
    #     print("[{}] {}".format(k, env))
    # print()
    # try:
    #     s = input("Choose an environment to run " + "(enter a number from 0 to {}): ".format(len(envs) - 1))
    #     # parse input into a number within range
    #     k = min(max(int(s), 0), len(envs))
    # except:
    #     k = 0
    #     print("Input is not valid. Use {} by default.\n".format(envs[k]))

    # Return the chosen environment name
    return envs


def choose_robots(exclude_bimanual=False):
    """
    Prints out robot options, and returns the requested robot. Restricts options to single-armed robots if
    @exclude_bimanual is set to True (False by default)

    Args:
        exclude_bimanual (bool): If set, excludes bimanual robots from the robot options

    Returns:
        str: Requested robot name
    """
    # Get the list of robots
    robots = {
        "Sawyer",
        "Panda",
        "Jaco",
        "Kinova3",
        "IIWA",
        "UR5e",
    }

    # Add Baxter if bimanual robots are not excluded
    if not exclude_bimanual:
        robots.add("Baxter")

    # Make sure set is deterministically sorted
    robots = sorted(robots)

    # Return requested robot
    return robots


def load_controller_config(custom_fpath=None, default_controller=None):
    """
    Utility function that loads the desired controller and returns the loaded configuration as a dict

    If @default_controller is specified, any value inputted to @custom_fpath is overridden and the default controller
    configuration is automatically loaded. See specific arg description below for available default controllers.

    Args:
        custom_fpath (str): Absolute filepath to the custom controller configuration .json file to be loaded
        default_controller (str): If specified, overrides @custom_fpath and loads a default configuration file for the
            specified controller.
            Choices are: {"JOINT_POSITION", "JOINT_TORQUE", "JOINT_VELOCITY", "OSC_POSITION", "OSC_POSE", "IK_POSE"}

    Returns:
        dict: Controller configuration

    Raises:
        AssertionError: [Unknown default controller name]
        AssertionError: [No controller specified]
    """
    # First check if default controller is not None; if it is not, load the appropriate controller
    if default_controller is not None:

        # Assert that requested default controller is in the available default controllers
        from robosuite.controllers import ALL_CONTROLLERS

        assert (
            default_controller in ALL_CONTROLLERS
        ), "Error: Unknown default controller specified. Requested {}, " "available controllers: {}".format(
            default_controller, list(ALL_CONTROLLERS)
        )

        # Store the default controller config fpath associated with the requested controller
        custom_fpath = os.path.join(
            os.path.dirname(__file__), "..", "controllers/config/{}.json".format(default_controller.lower())
        )

    # Assert that the fpath to load the controller is not empty
    assert custom_fpath is not None, "Error: Either custom_fpath or default_controller must be specified!"

    # Attempt to load the controller
    try:
        with open(custom_fpath) as f:
            controller_config = json.load(f)
    except FileNotFoundError:
        print("Error opening controller filepath at: {}. " "Please check filepath and try again.".format(custom_fpath))

    # Return the loaded controller
    return controller_config


if __name__ == "__main__":
    import json
    env = choose_mimicgen_environment()
    with open('mimicgen_env.json', 'w') as f:
        json.dump(env, f)
    robots = choose_robots()
    with open('mimicgen_robots.json', 'w') as f:
        json.dump(robots, f)
    controller_config = load_controller_config(default_controller="OSC_POSE")
    with open('mimicgen_controller_config.json', 'w') as f:
        json.dump(controller_config, f)

    options = {}
