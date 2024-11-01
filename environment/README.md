### `environment/`

Scripts and files saved here were/can be used to generate the Python environment employed for this study.

### Environment files
As [`conda`](https://docs.conda.io/projects/conda/en/latest/index.html) can be finicky generating environments across machines, we create environment files in various ways.
Hopefully, by providing all of this information, we can help aid replication.
- `env_cascades.yml`: generated in the standard way but may not work with different machines
- `env_cascades.txt`: meant to be explicit about which conda versions are utilized
- `env_from_history_cascades.yml`: according to [condas documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-an-environment-file-across-platforms), this is meant to be the most robust YAML file to work across different types of machines.

To create the virtual environment using one of the above files, you can try one of the below calls:
- `conda env create -f env_cascades.yml`
- `conda create --name cascades --file env_cascades.txt`
- `conda env create -f env_from_history_cascades.yml`

For more information, see the [`conda` documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

### Scripts
We utilized conda (V 24.4.0) to create the virtual environments that exported the above listed files.

First, we executed the below in our terminal.
```shell
conda create -n cascades python=3.12.2
conda activate cascades
```

Then, we ran the below `bash` scripts in the below order.

1. `generate_env.sh`: generated the final version of the python virtual environment
2. `create_env_yaml_files.sh`: generated the different environment files in this directory