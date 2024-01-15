import pandas as pd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def classification_evaluation():
    # Open answer.csv and create a DataFrame
    answer_df = pd.read_csv('/data/ephemeral/level2-klue-nlp-08/scripts/classification_evaluation/answer.csv')
    predict_df = pd.read_csv('/data/ephemeral/level2-klue-nlp-08/scripts/classification_evaluation/predict.csv')

    # Display the DataFrame
    true_labels = answer_df['label']
    predicted_labels = predict_df['pred_label']

    unique_labels = set(predict_df['pred_label'])
    labels = list(unique_labels)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Create a DataFrame for better visualization (optional)
    conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('/data/ephemeral/level2-klue-nlp-08/scripts/classification_evaluation/confusion_matrix.pdf', bbox_inches='tight')
    
if __name__ == '__main__':
    classification_evaluation()