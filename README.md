# Credit Score Classification
This project aims to aid financial institutions in reducing manual effort by classifying a person’s credit score into Poor, Standard, or Good based on their financial profile.

The classifier utilizes Random Forest along with thoughtful data processing and feature selection to enable faster and more accurate decisions.

## Project Highlights
- 100,000+ records with missing values and messy data — cleaned and processed gracefully
- Custom pipeline to fill missing values while retaining maximum information
- Feature selection with Random Forest — reducing from 28 to 14 key features, retaining nearly all predictive power (81% → 80%)
- Robust classifier with high performance (above 0.9 AUC for each class)
- An accessible Streamlit application for real-time credit score prediction
- Easily customizable and deployable pipeline for financial institutions

## Tech Stack
- Python (Pandas, Scikit-learn)
- Streamlit for UI
- Scikit-learn (Random Forest) for the classifier
- Matplotlib/Sseaborn for data visualization

## Installation
Clone the repository:
- git clone https://github.com/danisharain12/Credit-Score-Classification.git
- cd Credit-Score-Classification
- Create a virtual environment (optional but recommended):

- python -m venv venv
- source venv/bin/activate  # Mac or Linux
- venv\Scripts\activate     # Windows

## Installing prerequisites:
-- pip install -r requirements.txt

## How to Run Streamlit Application
- streamlit run app.py
- Then navigate to the Streamlit application in your browser (typically at http://localhost:8501).

## Deployment
- The application is deployed and can be accessed directly here:
- https://credit-score-classification1.streamlit.app/

## File Structure
└ Credit-Score-Classification/
   └ app.py
   └ data/
   └ notebooks/
   └ requirements.txt
   └ README.md
   
## Model Performance
- Random Forest classifier with above 0.9 AUC for each class (Poor, Standard, Good)
- Accuracy drops by only 1% after reducing number of features (81% → 80%)

## Contribution
- Contributions, bug reports, or suggestions for improvements are welcome!
- Please feel free to open an Issue or submit a Pull Request.

## Contact
- For questions or collaborations, connect with me on LinkedIn or through Github.

✅ Thank you for exploring this project!
✅ If you find it helpful, a (star) on the repository is much appreciated.

