import torch.nn as nn

class LogisticRegression(nn.Module):
    """
    A simple logistic regression model for classification.
    Given input features, it maps them to class probabilities.
    """
    
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        # Mapping from representation h to classes  
        self.model = nn.Linear(n_features, n_classes)

    def forward (self, x):
        return self.model(x)

