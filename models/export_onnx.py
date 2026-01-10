import torch
import torch.nn as nn
import json
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# ======== Fichiers ========
PTH_MODEL = "student_model_pro_20260108_223923.pth"
ONNX_MODEL = "student_model.onnx"
METADATA_JSON = "model_metadata.json"

# ======== Globals sécurisés ========
torch.serialization.add_safe_globals([IsotonicRegression, LogisticRegression])

# ======== Chargement checkpoint ========
checkpoint = torch.load(PTH_MODEL, map_location="cpu", weights_only=False)
print("✅ Checkpoint chargé")

FEATURE_COLUMNS = [
    'Niveau_etude',
    'Heures_etude_ordinal', 
    'Planning_ordinal',
    'Assiduite_ordinal',
    'Environnement_ordinal',
    'Sommeil_score',
    'Qualite_ordinal'
]

config_dict = checkpoint.get("model_config", {})
input_size = config_dict.get("input_size", 7)
hidden_sizes = tuple(config_dict.get("hidden_sizes", [16, 8]))
dropout_rate = config_dict.get("dropout_rate", 0.2)
normalization = config_dict.get("normalization", "layer")

print(f"Configuration du modèle : input={input_size}, hidden={hidden_sizes}, dropout={dropout_rate}, norm={normalization}")

# ======== Définition du modèle simple pour ONNX ========
class SimpleMLP(nn.Module):
    def __init__(self, input_size=7, hidden_sizes=(16, 8), dropout=0.2, normalization='layer'):
        super().__init__()
        layers = []
        current_size = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden))
            if normalization == 'layer':
                layers.append(nn.LayerNorm(hidden))
            elif normalization == 'batch':
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = hidden
        layers.append(nn.Linear(current_size, 1))  # output logits
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ======== Création et chargement du modèle ========
model = SimpleMLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout_rate, normalization=normalization)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ Modèle reconstruit et prêt pour export")

# ======== Test sur entrée dummy ========
with torch.no_grad():
    test_input = torch.randn(1, input_size) * 0.3 + 0.5
    test_output = model(test_input)
    probability = torch.sigmoid(test_output).item()
    threshold = checkpoint.get("training_results", {}).get("threshold_info", {}).get("optimal_threshold", 0.61)
    print(f"Test inference : logits={test_output.item():.4f}, probabilité={probability:.4f}, seuil={threshold:.2f}")

# ======== Export ONNX ========
try:
    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL,
        export_params=True,
        opset_version=17,  # Tract supporte très bien >=14
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        verbose=False
    )
    print(f"✅ Export ONNX réussi : {ONNX_MODEL} ({os.path.getsize(ONNX_MODEL)/1024/1024:.2f} MB)")
except Exception as e:
    print("❌ Erreur export ONNX:", e)
    import traceback
    traceback.print_exc()

# ======== Metadata pour Rust ========
metadata = {
    "model_info": {
        "format": "ONNX",
        "opset_version": 17,
        "input_shape": [1, input_size],
        "features": FEATURE_COLUMNS,
        "export_date": "2026-01-10"
    },
    "inference": {
        "threshold": threshold,
        "input_range": [0.0, 1.0],
        "output_type": "logits",
        "apply_sigmoid": True
    },
    "rust_config": {
        "input_name": "features",
        "output_name": "logits",
        "dtype": "f32",
        "requires_sigmoid": True
    }
}

with open(METADATA_JSON, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata JSON généré : {METADATA_JSON}")

print("\n✅ Processus terminé ! Modèle prêt pour inference Rust.")
