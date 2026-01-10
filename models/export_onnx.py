import torch
import torch.nn as nn
import torch.serialization
import json  # <-- IMPORT AJOUTÉ ICI
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# ========= PATHS =========
PTH_MODEL = "student_model_pro_20260108_223923.pth"
ONNX_MODEL = "student_model.onnx"

# ========= SÉCURITÉ PYTORCH 2.6+ =========
# Autoriser les globals nécessaires
torch.serialization.add_safe_globals([
    IsotonicRegression,
    LogisticRegression
])

# ========= LOAD CHECKPOINT =========
print("🔍 Chargement du checkpoint...")

# Charger avec weights_only=False
checkpoint = torch.load(PTH_MODEL, map_location="cpu", weights_only=False)

print("✅ Checkpoint chargé")

# ========= CONFIGURATION =========
# Définir les features manuellement pour éviter d'importer model.py
FEATURE_COLUMNS = [
    'Niveau_etude',
    'Heures_etude_ordinal', 
    'Planning_ordinal',
    'Assiduite_ordinal',
    'Environnement_ordinal',
    'Sommeil_score',
    'Qualite_ordinal'
]

# ========= RECONSTRUCTION DU MODÈLE =========
# Extraire la configuration
config_dict = checkpoint.get("model_config", {})
input_size = config_dict.get("input_size", 7)
hidden_sizes = tuple(config_dict.get("hidden_sizes", [16, 8]))
dropout_rate = config_dict.get("dropout_rate", 0.2)
normalization = config_dict.get("normalization", "layer")

print(f"🔧 Configuration extraite:")
print(f"   • Input size: {input_size}")
print(f"   • Hidden sizes: {hidden_sizes}")
print(f"   • Dropout: {dropout_rate}")
print(f"   • Normalization: {normalization}")

# ========= CRÉER LE MODÈLE SIMPLIFIÉ =========
class SimpleMLP(nn.Module):
    """Version simplifiée du modèle pour l'export ONNX"""
    def __init__(self, input_size=7, hidden_sizes=(16, 8), dropout=0.2, normalization='layer'):
        super(SimpleMLP, self).__init__()
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            
            if normalization == 'layer':
                layers.append(nn.LayerNorm(hidden_size))
            elif normalization == 'batch':
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Créer le modèle
model = SimpleMLP(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    dropout=dropout_rate,
    normalization=normalization
)

# Charger les poids
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ Modèle reconstruit")

# ========= VÉRIFICATION =========
print("\n🧪 Test du modèle...")
with torch.no_grad():
    # Test avec des valeurs réalistes [0, 1]
    test_input = torch.randn(1, input_size) * 0.3 + 0.5  # ~N(0.5, 0.3)
    test_output = model(test_input)
    probability = torch.sigmoid(test_output).item()
    
    # Trouver le seuil optimal
    threshold_info = checkpoint.get('training_results', {}).get('threshold_info', {})
    optimal_threshold = threshold_info.get('optimal_threshold', 0.61)
    
    print(f"   • Sortie brute: {test_output.item():.4f}")
    print(f"   • Probabilité: {probability:.4f}")
    print(f"   • Seuil optimal: {optimal_threshold:.3f}")
    print(f"   • Prédiction: {'RÉUSSITE' if probability >= optimal_threshold else 'ÉCHEC'}")

# ========= EXPORT ONNX =========
print(f"\n📤 Export ONNX vers: {ONNX_MODEL}")

try:
    dummy_input = torch.randn(1, input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        verbose=False
    )
    
    print("✅ Export ONNX réussi!")
    
    # ========= VÉRIFICATION FICHIER =========
    import os
    file_size = os.path.getsize(ONNX_MODEL) / 1024 / 1024
    print(f"📏 Taille fichier: {file_size:.2f} MB")
    
    # ========= MÉTADONNÉES POUR RUST =========
    metadata = {
        "model_info": {
            "format": "ONNX",
            "opset_version": 14,
            "input_shape": [1, input_size],
            "features": FEATURE_COLUMNS,
            "export_date": "2026-01-09"
        },
        "inference": {
            "threshold": optimal_threshold,
            "input_range": [0.0, 1.0],
            "output_type": "logits"
        },
        "rust_config": {
            "input_name": "features",
            "output_name": "logits",
            "dtype": "f32",
            "requires_sigmoid": True
        }
    }
    
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
   
    print(f"\n3. Seuil optimal: {optimal_threshold}")
    print("   • Calcul: probabilité = 1.0 / (1.0 + (-logits).exp())")
    print(f"   • Décision: probabilité >= {optimal_threshold} → RÉUSSITE")
    
except Exception as e:
    print(f"❌ Erreur export ONNX: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("✅ PROCESSUS TERMINÉ!")
print("="*50)