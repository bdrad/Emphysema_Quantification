# Emphysema_Quantification

This repository provides a pipeline for accurately quantifying emphysema score from chest CT scans.

See `requirements.txt` for required packages.

* `report_extraction.py` provides basic code for extracting emphysema extent classified by radiologist from a patient report.
* `data_processing_utils.py` contains a `wrapper_fn` that takes in a folder directory that contains a series of DICOM file representing one particular CT scan and obtains the emphysema from that CT scan. This function can be incorporated with multiprocessing that is not included here.
* `image_preprocessing_utils.py` and `emphysema_visualizer_utils.py` have some nice utility functions for visualizing 3D lung CT scans and emphysema.
* `train.py` provides code for determining cutoffs and associated metrics for emphysema classification into "none," "mild to moderate," and "severe" according to a training dataset of (emphysemas score, radiologist classification) pairs.
