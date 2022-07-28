# Stock price prediction

## Web Application

### Step 1: Install packages

1. Make sure pc is connected to internet
2. Open terminal and run `pip install virtualenv` to install virtualenv
3. Run `virtualenv --system-site-packages -p python ./venv` to create the virtual environment
4. Run `.\venv\Scripts\activate ` to start the virtual environment
5. Run `pip install -r requirements.txt` to install all required packages

### Step 2: Run application
1. In the project directory on terminal, run `cd app`
2. Then run `streamlit run app.py`, this automatically opens your browser and renders the application.

### How to add more dataset
To add more datasets for testing the model on the website, download the dataset, then place it inside the "app" folder, the dataset would then appear on the list of options on the web app.


## Jupyter notebook
1. Download anaconda from [anaconda_windows](https://repo.anaconda.com/archive/Anaconda3-2022.05-Windows-x86_64.exe) or [anaconda_mac](https://repo.anaconda.com/archive/Anaconda3-2022.05-MacOSX-x86_64.pkg)

2. Install and open the application

3. Open Jupyter notebook from the list of applications on the homepage

4. Open the the project.ipynb file in the notebook

5. Click on "Run" at the top bar, and select "Run All Cells" from the dropdown and it would execute and display the result of all the code.