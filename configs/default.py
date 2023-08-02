from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# MODEL
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.MODE = 'train'
_C.MODEL.RESUME = False
_C.MODEL.LAST_CKPT_FILE = ''
_C.MODEL.DETECTION = 'gt'

# -----------------------------------------------------------------------------
# Feature Extractor
_C.FE = CN()
_C.FE.INPUT_SIZE = ()
_C.FE.CHOICE = ""

# -----------------------------------------------------------------------------
# DATASET
_C.DATASET = CN()
_C.DATASET.DIR = ''
_C.DATASET.NAME = ''
_C.DATASET.SEQUENCE = []
_C.DATASET.CAMS = 0
_C.DATASET.TOTAL_FRAMES = 0
# -----------------------------------------------------------------------------
# SOLVER
_C.SOLVER = CN()
_C.SOLVER.TYPE = ''
_C.SOLVER.EPOCHS = 0
_C.SOLVER.EVAL_EPOCH = 0
_C.SOLVER.BATCH_SIZE = 0
_C.SOLVER.LR = 0.0
_C.SOLVER.MAX_PASSING_STEPS = 0
_C.SOLVER.W = 0
_C.SOLVER.W_TEST = 0
_C.SOLVER.FOCAL_ALPHA = 0.0
_C.SOLVER.FOCAL_GAMMA = 0
# -----------------------------------------------------------------------------
# OUTPUT
_C.OUTPUT = CN()
_C.OUTPUT.VISUALIZE = False
_C.OUTPUT.LOG = False
_C.OUTPUT.CKPT_DIR = ''
_C.OUTPUT.INFERENCE_DIR = ''
# -----------------------------------------------------------------------------
# TEST
_C.TEST = CN()
_C.TEST.CKPT_FILE_SG = ''
_C.TEST.CKPT_FILE_TG = ''
_C.TEST.FRAME_START = 0
_C.TEST.EDGE_THRESH = 0.0
# -----------------------------------------------------------------------------