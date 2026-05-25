# Architecture
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
FFN_DIM = 256
DROPOUT = 0.1
TIME_EMBED_L = 64
T_DIFF = 1000
MAX_SEQ_LEN = 72
N_FEATURES = 39

# Training
BATCH_SIZE = 64
PRETRAIN_LR = 1e-4
FINETUNE_LR = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PRETRAIN_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10
FINETUNE_MAX_EPOCHS = 200
FINETUNE_LR_RATIOS = [0.1, 0.3, 0.7, 1.0]  # per transformer layer; 0.0 = frozen

# ContiMask attribution
MASK_HIDDEN_DIM = 16
MASK_L = 12
MASK_TARGET_AREA = 0.1
MASK_EPOCHS = 200
PGPE_POP_SIZE = 50
LAMBDA_1 = 1
LAMBDA_2 = 0

# fmt: off
FEATURE_COLS = [
    # Vital signs
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    # Lab values
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
    # Demographics
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
]
# fmt: on
