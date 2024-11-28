import numpy as np
from sklearn.metrics import confusion_matrix

#from mfcc_extract import load_mfccs

class RecognitionEvaluation:
    def __init__(self, ground_truth, predicted):
    
        self.ground_truth = ground_truth.split() # load mfcc of original speech
        self.predicted = predicted.split() # predicted mfcc
        self.N = len(self.ground_truth)

        # Error counts
        self.substitution_errors = 0
        self.deletion_errors = 0
        self.insertion_errors = 0


    def compute_errors(self):
        gt_idx, pred_idx = 0, 0
        #print(self.ground_truth, self.predicted)
        while gt_idx < len(self.ground_truth) and pred_idx < len(self.predicted):
            if self.ground_truth[gt_idx] == self.predicted[pred_idx]:
                # No error, move both indices forward
                # print(ground_truth[gt_idx], predicted[pred_idx])
                gt_idx += 1
                pred_idx += 1
                
            else:
                # Substitution error
                self.substitution_errors += 1
                gt_idx += 1
                pred_idx += 1

        # Handle remaining ground truth words (deletions)
        while gt_idx < len(self.ground_truth):
            self.deletion_errors += 1
            gt_idx += 1

        # Handle remaining predicted words (insertions)
        while pred_idx < len(self.predicted):
            self.insertion_errors += 1
            pred_idx += 1

    def calculate_metrics(self):
    
        # Calculate performance metrics
        correct_recognition_percentage = 100 * (self.N - self.substitution_errors - self.deletion_errors) / self.N
        recognition_accuracy = 100 * (self.N - self.substitution_errors - self.deletion_errors - self.insertion_errors) / self.N
        error_rate = 100 * (self.substitution_errors + self.deletion_errors + self.insertion_errors) / self.N
        
        return {
            'Substitution Errors': self.substitution_errors,
            'Deletion Errors': self.deletion_errors,
            'Insertion Errors': self.insertion_errors,
            'Percentage Correct Recognition': correct_recognition_percentage,
            'Recognition Accuracy': recognition_accuracy,
            'Error Rate': error_rate,
            'Confusion Matrix': self._generate_confusion_matrix()
        }

    def _generate_confusion_matrix(self):
        # Generates and returns the confusion matrix for the ground truth vs predicted sequences.
        return confusion_matrix(self.ground_truth, self.predicted, labels=list(set(self.ground_truth + self.predicted)))

    def print_metrics(self):
        metrics = self.calculate_metrics()
        
        print(f"Substitution Errors: {metrics['Substitution Errors']}")
        print(f"Deletion Errors: {metrics['Deletion Errors']}")
        print(f"Insertion Errors: {metrics['Insertion Errors']}")
        print(f"Percentage of Correct Recognition: {metrics['Percentage Correct Recognition']:.2f}%")
        print(f"Recognition Accuracy: {metrics['Recognition Accuracy']:.2f}%")
        print(f"Error Rate: {metrics['Error Rate']:.2f}%")
        
        print("\nConfusion Matrix:")
        print(metrics['Confusion Matrix'])





if __name__ == "__main__":
    ground_truth = "heed hid head had hard hud hod hoard hood who'd heard"
    predicted = "heed hid head hard hud hoard hood heard pad wood head"

    evaluator = RecognitionEvaluation(ground_truth, predicted)
    evaluator.compute_errors()  # Compute errors
    evaluator.print_metrics()   # Print the evaluation results
