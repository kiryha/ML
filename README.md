# Rebelway ML Course
Machine Learning with Rebelway

## System Setup (Windows OS)

Install WSL: https://learn.microsoft.com/en-us/windows/wsl/install

"Turn Windows features on or off" turn ON:

- Virtual Machine Platform
- Windows Subsystem for Linux

Run "Ubuntu" and set user_name@password = kiryha@enter

Install Miniconda:
https://docs.anaconda.com/free/miniconda/

### Ubuntu Miniconda Environments 

Get existing environments: `conda info --envs`  
Create new environment: `conda create -n EnvironmentName python==3.10`
Activate environment: `conda activate EnvironmentName`

### Jupyter Notebook
From the "base" environment type "jupyter notebook" in Ubuntu, copy/paste URL to chrome, you will get your Notebook.

``path = "/home/kiryha/rebelway_pandas.ipynb``

### Houdini Environment
Edit houdini.env file, add (or update Eve launcher):
`PYTHONPATH = "//wsl.localhost/Ubuntu/home/kiryha/miniconda3/envs/houdini/lib/python3.10/site-packages"`