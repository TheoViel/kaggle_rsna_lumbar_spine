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

# NOISY_STUDIES = [
#     1085426528,
#     1395773918,
#     1879696087,
#     2135829458,
#     2388577668,
#     2662989538,
#     3495818564,
# ]

NOISY_STUDIES = [
    783154228, 520900899, 1217004843, 3828017267, 3369277408,
    1743493727, 1301627154, 4201106871, 3713534743,  413910863,
    105895264, 3469376405, 1088270559, 1289563234,  808294521,
    2638691430, 1314603564, 4266523380, 1891482189, 1179643011,
    2668759897, 3008676218,  757619082,  901299313,  704573554,
    4072191052, 189360935, 2238966046,  959290081, 1106510276,
    1666601651, 2966328820, 1459964234, 2621581337, 3996069892,
    341051344, 305152236, 1791596037, 3968285352, 2991382385,
    4172077685, 3740680860,   58813022, 2046176090, 1548005561,
    3515641631, 3227154093, 4058604433, 1871675162, 3781188430,
    3201694970, 3337564969, 1972129014,  504362668, 4259049254,
    1755159626, 3029953735, 1353517692,  497870715,  325485990,
    3192842688, 264945797, 1176604093,  677879566,  796739553,
    885894528, 3507369254, 3617361428, 1697944783, 2059107661,
    1647904243, 4279881930,  618246392, 2797118205,  296083289,
    1360027517, 1639920408, 1140449293, 3587452388,  976356113,
    1199116491, 3496128487, 1075863395, 2565951228, 1436167447,
    2530679352, 1205664021, 2239199413, 3847808992, 1670838975,
    480042730, 1460690973, 3418118075, 3646569986,  344297746,
    1009905322, 3718047621, 3328458132,  679759364, 1009445512
]