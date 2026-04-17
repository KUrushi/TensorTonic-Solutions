import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    if num_classes is None:
        num_classes = max(y_true) + 1
        
    cm = np.zeros((num_classes,num_classes))
    for y, p in zip(y_true, y_pred):
        cm[y,p] += 1

    match normalize:
        case 'none':
            return cm
        case 'all':
            return cm / cm.sum()
        case 'true':
            return cm / cm.sum(axis=1, keepdims=True)
        case 'pred':
            return cm / cm.sum(axis=0, keepdims=True)
        
        