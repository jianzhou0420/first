import argparse
from typing import List, Optional, Union

import yacs.config

# Default config node


class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config


CONFIG_FILE_SEPARATOR = ';'

_C = CN()
_C.CMD_TRAILING_OPTS = []
# --------------------------------------------------------------------------
# TRAIN_DATASET
_C.TRAIN_DATASET = CN()
# --------------------------------------------------------------------------


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.CMD_TRAILING_OPTS = []

    config.freeze()
    return config


def build_args(path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=False,
        help="path to config yaml containing info about experiment",
        default='./zero/expForwardKinematics/config/expBase_Lotus.yaml',
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (use , to separate values in a list)",
    )

    args = parser.parse_args()

    if path is not None:
        args.exp_config = path
    config = get_config(args.exp_config, args.opts)
    return config
