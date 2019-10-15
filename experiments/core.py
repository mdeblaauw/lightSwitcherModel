import torch

def create_nshot_task_label(k, q):
    """Creates an n-shot task label.
    Label has the structure:
        [0]*q + [1]*q + ... + [k-1]*q
    # Arguments
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task
    # Returns
        y: Label vector for n-shot task of shape [q * k, ]
    """
    y = torch.arange(0, k, 1 / q).long()
    return(y)

def prepare_nshot_task(batch, k, q):
    x, y = batch
    x = x.float()
    # Create dummy 0-(num_classes - 1) label
    y = create_nshot_task_label(k, q)
    return(x, y)

def categorical_accuracy(y, y_pred):
    return(torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0])