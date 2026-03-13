\# MLflow Lab — Titanic Survival Prediction



This lab demonstrates MLflow experiment tracking using the Titanic dataset.  

Three models are trained and compared: Logistic Regression, Random Forest, and Gradient Boosting.



\## Modifications from original lab

\- Dataset: Titanic (vs. original lab dataset)

\- Models: LogisticRegression, RandomForest, GradientBoosting (3-way comparison)

\- Metrics tracked: Accuracy, Precision, Recall, F1 Score

\- Preprocessing pipeline included in training script



\## Project Structure

```

mlflow-titanic/

├── data/

│   └── titanic.csv

├── train.py

├── requirements.txt

└── README.md

```



\## Setup \& Run



1\. \*\*Clone the repo and install dependencies\*\*

```bash

pip install -r requirements.txt

```



2\. \*\*Download the Titanic dataset\*\*  

&#x20;  Get `train.csv` from \[Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data),  

&#x20;  rename it to `titanic.csv`, and place it in the `data/` folder.



3\. \*\*Run the training script\*\*

```bash

python train.py

```



4\. \*\*Launch the MLflow UI\*\*

```bash

mlflow ui

```

Then open \[http://localhost:5000](http://localhost:5000) in your browser to compare runs.



\## Results

Three runs are logged under the experiment `titanic-survival-prediction`, each tracking:

\- `accuracy`

\- `precision`

\- `recall`

\- `f1\_score`



Use the MLflow UI to compare model performance across all runs.

