import os
import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
from torch.autograd import Variable
from evaluationScript.eval import nodule_evaluation
from evaluationScript.logs import setup_logging


class ModelEvaluator:
    def __init__(self,rcnn = False):
        """
        Initialize the ModelEvaluator class.

        Parameters:
        - net: The neural network model.
        - eval_loader: DataLoader for the evaluation dataset.
        - save_dir: Directory where the results will be saved.
        """
        self._rpn_submission_path = None  # Private attribute
        self._eval_dir = None  # Private attribute

    def eval(self, net, eval_loader, save_dir,epoch):
        """
        Evaluates the network on the evaluation dataset and generates a CSV file
        with the region proposal network (RPN) predictions.

        Parameters:
        - epoch: Current epoch number (for logging).

        Returns:
        - Path to the evaluation directory.
        """
        # Set model to evaluation mode
        net.set_mode('eval')

        # To store RPN results
        rpn_res = []

        # Iterate over the evaluation dataset
        for j, (input, truth_box, truth_label) in tqdm(enumerate(eval_loader), total=len(eval_loader), desc='Eval %d' % epoch):
            with torch.no_grad():  # Disable gradient computation for evaluation
                # Move input to GPU
                input = Variable(input).cuda()
                truth_box = np.array(truth_box)
                truth_label = np.array(truth_label)

                # Use mixed precision for inference
                with torch.cuda.amp.autocast():
                    net.forward(input, truth_box, truth_label)

                # Extract RPN proposals and process the coordinates
                rpns = net.rpn_proposals.cpu().numpy()
                rpns = rpns[:, 1:]  # Exclude the first column
                rpns = rpns[:, [3, 2, 1, 6, 5, 4, 0]]  # Rearrange the columns as needed

                # Get the filename for the current input
                pid = eval_loader.dataset.filenames[j]
                names = np.array([[pid]] * len(rpns))  # Create a column with the patient ID

                # Debugging shape info
                print("Shape of names:", names.shape)
                print("Shape of rpns after processing:", rpns.shape)

                # Concatenate the names and RPN results
                rpn_res.append(np.concatenate([names, rpns], axis=1))

        # Concatenate all RPN results into a single array
        rpn_res = np.concatenate(rpn_res, axis=0)

        # Define column names for the CSV file
        col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'probability']

        # Create directory to save the evaluation results
        self._eval_dir = os.path.join(save_dir, 'FROC')
        if not os.path.exists(self._eval_dir):
            os.makedirs(self._eval_dir)

        # Define the path for the RPN submission CSV file
        self._rpn_submission_path = os.path.join(self._eval_dir, 'submission_rpn.csv')

        # Save the RPN results as a CSV file
        df = pd.DataFrame(rpn_res, columns=col_names)
        df.to_csv(self._rpn_submission_path, index=False)
        torch.cuda.empty_cache()
        
    def eval_all(self, net, eval_loader, save_dir,epoch):
        """
        Evaluates the network on the evaluation dataset and generates a CSV file
        with the region proposal network (RPN) predictions.

        Parameters:
        - epoch: Current epoch number (for logging).

        Returns:
        - Path to the evaluation directory.
        """
        # Set model to evaluation mode
        net.set_mode('eval')

        # To store RPN results
        rpn_res = []
        rcnn_res = []
        ensemble_res = []

        # Iterate over the evaluation dataset
        for j, (input, truth_box, truth_label) in tqdm(enumerate(eval_loader), total=len(eval_loader), desc='Eval %d' % epoch):
            with torch.no_grad():  # Disable gradient computation for evaluation
                # Move input to GPU
                input = Variable(input).cuda()
                truth_box = np.array(truth_box)
                truth_label = np.array(truth_label)

                # Use mixed precision for inference
                with torch.cuda.amp.autocast():
                    net.forward(input, truth_box, truth_label)

                # Extract RPN proposals and process the coordinates
                rpns = net.rpn_proposals.cpu().numpy()
                rcnns = net.rcnns.cpu().numpy()
                ensembles = net.ensemble_proposals.cpu().numpy()

                rpns = rpns[:, 1:]  # Exclude the first column
                rpns = rpns[:, [3, 2, 1, 6, 5, 4, 0]]  # Rearrange the columns as needed

                rcnns = rcnns[:, 1:-1]
                rcnns = rcnns[:, [3, 2, 1, 6, 5, 4, 0]]

                ensembles = ensembles[:, 1:]
                ensembles = ensembles[:, [3, 2, 1, 6, 5, 4, 0]]
                
                # Get the filename for the current input
                pid = eval_loader.dataset.filenames[j]
                names = np.array([[pid]] * len(rpns))  # Create a column with the patient ID

                # Concatenate the names and RPN results
                rpn_res.append(np.concatenate([names, rpns], axis=1))
                rcnn_res.append(np.concatenate([names, rcnns], axis=1))
                ensemble_res.append(np.concatenate([names, ensembles], axis=1))
        # Concatenate all RPN results into a single array
        rpn_res = np.concatenate(rpn_res, axis=0)
        rcnn_res = np.concatenate(rcnn_res, axis=0)
        ensemble_res = np.concatenate(ensemble_res, axis=0)
        # Define column names for the CSV file
        col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'probability']

        # Create directory to save the evaluation results
        self._eval_dir = os.path.join(save_dir, 'FROC')
        if not os.path.exists(self._eval_dir):
            os.makedirs(self._eval_dir)

        # Define the path for the RPN submission CSV file
        self._rpn_submission_path = os.path.join(self._eval_dir, 'submission_rpn.csv')
        self._rcnn_submission_path = os.path.join(self._eval_dir, 'submission_rcnn.csv')
        self._ensemble_submission_path = os.path.join(self._eval_dir, 'submission_ensemble.csv')

        # Save the RPN results as a CSV file
        df = pd.DataFrame(rpn_res, columns=col_names)
        df.to_csv(self._rpn_submission_path, index=False)

        df = pd.DataFrame(rcnn_res, columns=col_names)
        df.to_csv(self._rcnn_submission_path, index=False)

        df = pd.DataFrame(ensemble_res, columns=col_names)
        df.to_csv(self._ensemble_submission_path, index=False)
        torch.cuda.empty_cache()

    
    def froc(self, epoch, writer, test_annotation_dir, series_uids_path,save_filename):
        """
        Compute the Free-Response Operating Characteristic (FROC) and log the sensitivities
        at various false positive rates using TensorBoard.

        Parameters:
        - epoch: Current epoch number (for logging).
        - writer: TensorBoard writer for logging metrics.
        - test_annotation_dir: Directory containing ground truth annotations.
        - series_uids_path: Path to the series UIDs file.
        """
        # Check if eval() has been called and paths are available
        if self._rpn_submission_path is None or self._eval_dir is None:
            raise ValueError("Evaluation has not been performed. Please run eval() before froc().")

        # Compute recall at different false positive rates (FPR)
        recalls, mean_recall = self.iou_FROC(test_annotation_dir, series_uids_path, self._rpn_submission_path, os.path.join(self._eval_dir, 'iou/rpn'+save_filename), fixed_prob_threshold=0.5)

        # Log the sensitivity at each FPR threshold to TensorBoard
        fpr_thresholds = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        for i, fpr in enumerate(fpr_thresholds):
            writer.add_scalar(f'sens_{fpr}', recalls[i], epoch)

        # Log the mean recall (average sensitivity) to TensorBoard
        writer.add_scalar('sens_mean', mean_recall, epoch)

        # Clear CUDA memory to avoid memor  y leaks
        torch.cuda.empty_cache()

    def iou_FROC(self,annot_path, series_uids_path, pred_results_path, output_dir, iou_threshold=0.1, fixed_prob_threshold=0.5):
        FP_ratios = [0.125, 0.25, 0.5, 1, 2, 4, 8.]
        setup_logging(level='info', log_file=os.path.join(output_dir, 'log.txt'))  
        froc_out, fixed_out, (best_f1_score, best_f1_threshold),fps2_sens = nodule_evaluation(annot_path = annot_path,
                                                                                    series_uids_path = series_uids_path, 
                                                                                    pred_results_path = pred_results_path,
                                                                                    output_dir = output_dir,
                                                                                    iou_threshold = iou_threshold,
                                                                                    fixed_prob_threshold=fixed_prob_threshold)
        fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points = froc_out
        fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score = fixed_out
        metrics = {'tp': fixed_tp,
                    'fp': fixed_fp,
                    'fn': fixed_fn,
                    'recall': fixed_recall,
                    'precision': fixed_precision,
                    'f1_score': fixed_f1_score,
                    'best_f1_score': best_f1_score,
                    'best_f1_threshold': best_f1_threshold}
        mean_recall = np.mean(np.array(sens_points))
        metrics['froc_mean_recall'] = float(mean_recall)
        for fp_ratio, sens_p in zip(FP_ratios, sens_points):
            metrics['froc_{}_recall'.format(str(fp_ratio))] = sens_p
        return sens_points, mean_recall
# Usage:
# Initialize the evaluator
# evaluator = ModelEvaluator(net, eval_loader, save_dir)

# Perform evaluation and get eval_dir
# eval_dir = evaluator.eval(epoch)

# Compute FROC after evaluation
# evaluator.froc(epoch, writer, test_annotation_dir, series_uids_path)
