import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000") # setting URI to the link, previous error was given cause confusion matrix was a file and hot http/https, this was a bug from MLflow

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 5

# Mention the experiment below
mlflow.autolog()
mlflow.set_experiment('MLOPS-Exp1')

with mlflow.start_run(): # Start mlflow
    # below this we start writing our training code 
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state = 42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # Saveplot
    plt.savefig("Confusion-matrix.png")

    # Log artifacts using MLflow
    mlflow.log_artifact(__file__) # To log the file we are writing the script in

    # Tags
    # mlflow.set_tag # For a single key-value pair
    mlflow.set_tags({"Author": "Dev", "Project": "Wine Classification"}) # Give and entire dictionary

    print(accuracy)