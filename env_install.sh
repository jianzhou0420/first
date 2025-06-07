mamba env create -f -y conda_env,yaml
mamba activate equidiff

pip uninstall robosuite
cd third_party/robosuite
pip install -e .
python -c "import robosuite; print(robosuite.__file__)"

cd ../robosuite-task-zoo
pip install -e .
python -c "import robosuite_task_zoo; print(robosuite_task_zoo.__file__)"

cd ../robomimic
pip install -e .
python -c "import robomimic; print(robomimic.__file__)"

cd ../mimicgen
pip install -e .
python -c "import mimicgen; print(mimicgen.__file__)"
