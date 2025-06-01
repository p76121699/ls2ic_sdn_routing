# Configuration File Guidelines
If you would like to define your own algorithm configuration, please follow the format provided in this directory to ensure compatibility with the framework.

1. Algorithm Configuration
Use `ospf_config.py` as a reference for the minimal set of required parameters. The following fields are essential:

`algs_name`: This must match the filename of your algorithm in the `algs/` directory.

`action_dim`: This defines the number of available actions, typically corresponding to the number of top-k shortest paths in the routing scenario.

2. Environment Configuration
Environment-related parameters are all required. Please refer to the existing environment configs in this directory as templates for creating your own.