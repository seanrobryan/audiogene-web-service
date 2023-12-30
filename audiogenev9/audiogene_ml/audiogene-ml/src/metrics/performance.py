import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def unweighted_accuracy(y_true, y_test):
    """ Calculates the unweighted average of the per class accuracy."""
    matrix = confusion_matrix(y_true, y_test)
    
    TPs = matrix.diagonal()        # Diagonal count of correct classifications
    r_sum = np.sum(matrix, axis=1) # Count of instances per class
    
    return (1/matrix.shape[0])*(np.sum(np.divide(TPs, r_sum)))


def top_k_unweighted_accuracy(y_true, y_probs, k: int = 3, labels=None, encoder=None):
    """ Calculates the top k unweighted average of the per class accuracy.
    Calculated as the sum of unweighted accuracies for the top k predictions.
    """
    if labels is None:
        unique_classes = [x for x in range(y_probs.shape[1])]
    else:
        unique_classes = labels

    if encoder is not None:
        y_true = encoder.transform(y_true)

    top_k = np.argsort(y_probs)[:, ::-1][:, :k]
    res_df = pd.DataFrame(data=np.zeros((len(unique_classes), 2)), columns=['top_k_hits', 'total'], index=unique_classes)
    
    for c, tk in zip(y_true, top_k):
        res_df.loc[c, 'total'] = res_df.loc[c, 'total'] + 1
        if c in tk:
            res_df.loc[c, 'top_k_hits'] = res_df.loc[c, 'top_k_hits'] + 1

    res_df['acc'] = res_df['top_k_hits'] / res_df['total']
    return res_df['acc'].mean()
            

def compare_prediction_rate(conf_mat_disp: ConfusionMatrixDisplay) -> pd.DataFrame:
    """ Counts the instances of each class and the number of times it was predicted."""
    
    # Get the confusion matrix array and labels from the display object
    matrix = conf_mat_disp.confusion_matrix
    labels = conf_mat_disp.display_labels
    
    r_sum = np.sum(matrix, axis=0) # Count of instances per class
    c_sum = np.sum(matrix, axis=1) # Count of predictions per class
    return pd.DataFrame(np.column_stack([r_sum, c_sum]), columns=['Num Instances', 'Times Predicted'], index=labels)
    