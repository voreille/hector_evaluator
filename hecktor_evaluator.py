import os
import zipfile
from pathlib import Path
import warnings
from shutil import rmtree
import time

import pandas as pd
import numpy as np
import SimpleITK as sitk

from utils import compute_scores, get_np_volume_from_sitk, resample


class HecktorEvaluator:
    def __init__(self,
                 ground_truth_folder,
                 bounding_boxes_filepath,
                 extraction_folder="data/extraction/",
                 round_number=1):
        """Evaluator for the Hecktor Challenge

        Args:
            ground_truth_folder (str): the path to the folder 
                                       containing the ground truth segmentation.
            bounding_boxes_filepath (str): the path to the csv file which defines
                                           the bounding boxes for each patient.
            extraction_folder (str, optional): the path to the folder where the 
                                               extraction of the .zip submission 
                                               will take place. Defaults to "data/tmp/".
                                               This folder has to be created beforehand.
            round_number (int, optional): the round number. Defaults to 1.
        """
        self.groud_truth_folder = Path(ground_truth_folder)
        self.round = round_number
        self.extraction_folder = Path(extraction_folder)
        self.bounding_boxes_filepath = Path(bounding_boxes_filepath)

    def _evaluate(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
            - submission_file_path : local file path of the submitted file
            - aicrowd_submission_id : A unique id representing the submission
            - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]
        aicrowd_submission_id = client_payload["aicrowd_submission_id"]
        aicrowd_participant_uid = client_payload["aicrowd_participant_id"]
        submission_extraction_folder = self.extraction_folder / (
            'submission' + str(aicrowd_submission_id) + '/')

        submission_extraction_folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(Path(submission_file_path).resolve()),
                             "r") as zip_ref:
            zip_ref.extractall(str(submission_extraction_folder.resolve()))

        groundtruth_paths = [
            f for f in self.groud_truth_folder.rglob("*.nii.gz")
        ]
        bb_df = pd.read_csv(str(
            self.bounding_boxes_filepath.resolve())).set_index("PatientID")

        score_columns = ["dice_score", "sensitivity", "precision"]
        results_df = pd.DataFrame(columns=score_columns)

        missing_patients = list()
        unresampled_patients = list()
        for path in groundtruth_paths:
            patient_id = path.name.split(".")[0]
            prediction_files = [
                f for f in submission_extraction_folder.rglob(patient_id +
                                                              "*.nii.gz")
            ]
            if len(prediction_files) > 1:
                raise Exception(
                    "There is too many prediction files for patient {}".format(
                        patient_id))
            elif len(prediction_files) == 0:
                results_df = results_df.append(
                    {
                        "dice_score": 0,
                        "sensitivity": 0,
                        "precision": 0,
                    },
                    ignore_index=True)

                missing_patients.append(patient_id)
                continue

            sitk_image_gt = sitk.ReadImage(str(path.resolve()))
            sitk_image_pred = sitk.ReadImage(str(
                prediction_files[0].resolve()))

            np_volume_pred, px_spacing_pred, origin_pred = get_np_volume_from_sitk(
                sitk_image_pred)
            np_volume_gt, px_spacing_gt, origin_gt = get_np_volume_from_sitk(
                sitk_image_gt)

            bb = (bb_df.loc[patient_id, "x1"], bb_df.loc[patient_id, "y1"],
                  bb_df.loc[patient_id, "z1"], bb_df.loc[patient_id, "x2"],
                  bb_df.loc[patient_id, "y2"], bb_df.loc[patient_id, "z2"])

            if sitk_image_gt.GetSpacing() != sitk_image_pred.GetSpacing():
                unresampled_patients.append(patient_id)

            # Crop to the bonding box and/or resample to the original spacing
            np_volume_pred = resample(np_volume_pred, origin_pred,
                                      px_spacing_pred, px_spacing_gt, bb)
            np_volume_gt = resample(np_volume_gt, origin_gt, px_spacing_gt,
                                    px_spacing_gt, bb)

            dice_score, sensitivity, precision = compute_scores(
                np_volume_gt, np_volume_pred)
            results_df = results_df.append(
                {
                    "dice_score": dice_score,
                    "sensitivity": sensitivity,
                    "precision": precision,
                },
                ignore_index=True)

        _result_object = dict(results_df.mean())

        rmtree(str(submission_extraction_folder.resolve()))
        messages = list()
        if len(unresampled_patients) > 0:
            messages.append(
                "The following patient(s) was/were not resampled back"
                " to the original resolution: {patients}."
                "\nWe applied a nearest neighbor resampling.\n".format(
                    patients=unresampled_patients))

        if len(missing_patients) > 0:
            messages.append(
                "The following patient(s) was/were missing: {patients}."
                "\nA score of 0 was attributed to them".format(
                    patients=missing_patients))
        _result_object["message"] = "".join(messages)
        return _result_object


if __name__ == "__main__":
    ground_truth_folder = "data/ground_truth/"
    bounding_boxes_file = "data/bboxes.csv"
    _client_payload = {}
    _client_payload[
        "submission_file_path"] = "data/sample_submission_missing.zip"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

    # Instantiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = HecktorEvaluator(ground_truth_folder,
                                         bounding_boxes_file)
    # Evaluate
    start = time.process_time()
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print("Time to compute the sample {}".format(time.process_time() - start))
    print(result)
    print(result["message"])
