mamba env create -f conda_env,yaml
mamba activate equidiff

pip uninstall robosuite
cd third_party/robosuite
pip install -e .
cd ../..
python -c "import robosuite; print(robosuite.__file__)"
