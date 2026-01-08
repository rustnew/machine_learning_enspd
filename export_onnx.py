import torch
import torch.nn as nn
from model import StudentSuccessModel, FEATURE_COLUMNS

# ========= PATHS =========
PTH_MODEL = "student_model_pro_20260108_223923.pth"
ONNX_MODEL = "export.onnx"

# ========= LOAD CHECKPOINT =========
checkpoint = torch.load(PTH_MODEL, map_location="cpu")

model = StudentSuccessModel(
    input_size=len(FEATURE_COLUMNS),
    dropout=checkpoint["config"]["dropout_rate"],
    normalization=checkpoint["config"]["normalization"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ========= DUMMY INPUT =========
dummy_input = torch.randn(1, len(FEATURE_COLUMNS))

# ========= EXPORT =========
torch.onnx.export(
    model,
    dummy_input,
    ONNX_MODEL,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("✅ ONNX généré avec succès :", ONNX_MODEL)
