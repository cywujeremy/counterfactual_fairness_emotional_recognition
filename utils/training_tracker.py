import matplotlib.pyplot as plt

class TrainingTracker:

    def __init__(self, experiment_name):

        self.experiment_name = experiment_name

        self.training_loss = []
        self.training_uar = []
        self.training_accuracy = []
        self.validation_confusion_matrix = []
        self.validation_loss = []
        self.validation_uar = []
        self.validation_accuracy = []

        self.best_epoch = 0
        self.best_model = None
    
