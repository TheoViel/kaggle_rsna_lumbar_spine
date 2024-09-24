NUM_WORKERS = 8

DATA_PATH = "../input/"
LOG_PATH = "../logs/"
OUT_PATH = "../output/"

DEVICE = "cuda"

NEPTUNE_PROJECT = "KagglingTheo/RSNA-Lumbar-Spine-Classification"

MODES = {"T2_Sagittal": "scs", "T1_Sagittal": "nfn", "T2_Axial": "ss"}

SEVERITIES = ["Normal/Mild", "Moderate", "Severe"]

CLASSES_SEG = [
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]

CLASSES_SCS = [
    "spinal_canal_stenosis_l1_l2",
    "spinal_canal_stenosis_l2_l3",
    "spinal_canal_stenosis_l3_l4",
    "spinal_canal_stenosis_l4_l5",
    "spinal_canal_stenosis_l5_s1",
]

CLASSES_NFN = [
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
]

CLASSES_NFN_NOSIDE = [
    "neural_foraminal_narrowing_l1_l2",
    "neural_foraminal_narrowing_l2_l3",
    "neural_foraminal_narrowing_l3_l4",
    "neural_foraminal_narrowing_l4_l5",
    "neural_foraminal_narrowing_l5_s1",
]

CLASSES_SS = [
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

CLASSES_SS_NOLVL = [
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]

CLASSES_CROP = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
LEVELS_ = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

CLASSES = CLASSES_SCS + CLASSES_NFN + CLASSES_SS

NOISY_SERIES = [
    1518511736,
    2278678071,
    3230157587,
    1032434193,
    692927423,
    3802667261,
    856763877,
]

NOISY_STUDIES = [
    1085426528,
    1395773918,
    1879696087,
    2135829458,
    2388577668,
    2662989538,
    3495818564,
]
