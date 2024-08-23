# To update installed packages
python -m pip freeze > requirements.txt

# To insall required packages
python -m pip install -r requirements.txt

# To start a new venv in the local directory
python -m venv ./.venv

# To change into venv in terminal
source .venv/bin/activate
