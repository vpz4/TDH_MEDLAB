# TDH

The Tabular Data Harmonization (TDH) tool is a Flask-based application developed in Python 3.9 and designed to facilitate metadata extraction, semantic matching, and final dataset harmonization. It supports input in `.csv`, `.xlsx`, or `.json` formats and produces structured reports summarizing metadata and feature-level diagnostics.

## Folder Structure
- `results/` - Directory for storing generated reports and harmonized datasets.
- `static/images/` - Contains logos used in the HTML interface.
- `templates/` - Stores the HTML script (`index.html`) along with branding logos.
- `app.py` - Main Flask application script.
- `Dockerfile` - Configuration for containerizing the application.
- `README.md` - Documentation for installation and usage.
- `requirements.txt` - Lists required Python dependencies.

## Input & Output
# Functionality 1 - Extraction of the metadata harmonization report
- **Input:** 
  - A data quality evaluation report (as extracted by the TDC: https://github.com/vpz4/TDC_MEDLAB) in `.xlsx`format.
  - A data model in `.XML` or `.OWL` format.
- **Output:**
  - **Data harmonization report** - Matched features, value ranges, IDs.
# Functionality 2 - Final harmonization process 
- **Input:** 
  - The metadata harmonization report (as extracted by Functionality 1) in `.xlsx`format.
  - The original tabular dataset in `.xlsx` or `.csv` format.
- **Output:**
  - **Data harmonization report** - Harmonized dataset.

## Functionalities
- To be added

## Installation & Execution
### Clone the Repository
```bash
git clone https://github.com/vpz4/TDH_MEDLAB.git
cd MHDT
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the Application
```bash
python app.py
```
Access the web interface at `http://127.0.0.1:5000/main`

## Docker Deployment
### Build and Run the Container
```bash
docker build -t tdh_medlab-app .
docker run -d -p 5000:5000 -v C/TDH/results:/app/results --name tdh_medlab-app tdh_medlab-app
```
### Access the Application
Navigate to `http://127.0.0.1:5000/main`

## References & Publications
- 

## License
This project is released under the MIT License.