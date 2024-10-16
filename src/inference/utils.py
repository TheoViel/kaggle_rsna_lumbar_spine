import numpy as np
import pandas as pd


def sub_to_dict(file):
    """
    Converts a submission file to a dict with predictions.

    Args:
        truths (str): Path to submission file.

    Returns:
        dict: Dictionary containing {study: predictions} pairs.
    """
    sub = pd.read_csv(file)
    sub['study'] = sub['row_id'].apply(lambda x: x.split('_')[0])
    sub['tgt'] = sub['row_id'].apply(lambda x: x.split('_', 1)[-1])

    cols = [
        "spinal_canal_stenosis_l1_l2",
        "spinal_canal_stenosis_l2_l3",
        "spinal_canal_stenosis_l3_l4",
        "spinal_canal_stenosis_l4_l5",
        "spinal_canal_stenosis_l5_s1",
        "left_neural_foraminal_narrowing_l1_l2",
        "left_neural_foraminal_narrowing_l2_l3",
        "left_neural_foraminal_narrowing_l3_l4",
        "left_neural_foraminal_narrowing_l4_l5",
        "left_neural_foraminal_narrowing_l5_s1",
        "right_neural_foraminal_narrowing_l1_l2",
        "right_neural_foraminal_narrowing_l2_l3",
        "right_neural_foraminal_narrowing_l3_l4",
        "right_neural_foraminal_narrowing_l4_l5",
        "right_neural_foraminal_narrowing_l5_s1",
        "left_subarticular_stenosis_l1_l2",
        "left_subarticular_stenosis_l2_l3",
        "left_subarticular_stenosis_l3_l4",
        "left_subarticular_stenosis_l4_l5",
        "left_subarticular_stenosis_l5_s1",
        "right_subarticular_stenosis_l1_l2",
        "right_subarticular_stenosis_l2_l3",
        "right_subarticular_stenosis_l3_l4",
        "right_subarticular_stenosis_l4_l5",
        "right_subarticular_stenosis_l5_s1",
    ]

    preds_dict = {}
    for study, dfs in sub.groupby('study'):
        # display(dfs)
        dfs = dfs[["tgt", "normal_mild", "moderate", "severe"]].set_index("tgt")
        preds = np.zeros((25, 3))
        for i, c in enumerate(cols):
            preds[i] = dfs.loc[c].values
        preds_dict[int(study)] = preds
    return preds_dict
