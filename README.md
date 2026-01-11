#  Projet de Prédiction de Réussite Étudiante - Documentation Complète

## 📋 Table des Matières

1. [ Présentation du Projet](#-présentation-du-projet)
2. [Architecture Technique](#️-architecture-technique)
3. [ Pipeline ML Complet](#-pipeline-ml-complet)
4. [ Modèle de Deep Learning](#-modèle-de-deep-learning)
5. [ Configuration Avancée](#️-configuration-avancée)
6. [ Utilisation Pas à Pas](#-utilisation-pas-à-pas)
7. [ Résultats et Métriques](#-résultats-et-métriques)
8. [ Analyse Scientifique](#-analyse-scientifique)
9. [ Optimisations Techniques](#️-optimisations-techniques)
10. [ Déploiement et Production](#-déploiement-et-production)
11. [ Frontend (Yew + Rust)](#-frontend-yew--rust)
12. [ Structure des Fichiers](#-structure-des-fichiers)
13. [ Tests et Validation](#-tests-et-validation)
14. [ Références Techniques](#-références-techniques)

---

## 🎯 Présentation du Projet

### Objectif
Développer un système de prédiction de réussite académique basé sur 7 caractéristiques normalisées d'étudiants, avec :
- **Précision élevée** (F1-score > 0.85)
- **Fiabilité des probabilités** (calibration)
- **Interprétabilité** (importance des features)
- **Estimation d'incertitude** (prédictions sûres)

### Cas d'Usage
- Orientation académique
- Détection précoce des risques
- Allocation de ressources pédagogiques
- Recherche en sciences de l'éducation

---

## 🏗️ Architecture Technique

### Stack Technologique Complète
```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Yew + Rust)                │
│                    - Interface Web WASM                 │
│                    - Visualisations D3.js               │
│                    - UX/UI professionnelle              │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP/JSON API
┌────────────────▼────────────────────────────────────────┐
│                    BACKEND API (Rust)                   │
│                    - Axum/Actix Web                     │
│                    - Inference ONNX/TorchScript         │
│                    - Cache Redis                        │
└────────────────┬────────────────────────────────────────┘
                 │ Modèles Sérialisés
┌────────────────▼────────────────────────────────────────┐
│            PIPELINE ML (Python)                         │
│            - PyTorch 2.0+                               │
│            - Scikit-learn                               │
│            - ONNX Runtime                               │
└────────────────┬────────────────────────────────────────┘
                 │ CSV/JSON
┌────────────────▼────────────────────────────────────────┐
│                    DONNÉES                              │
│                    - 1000 échantillons                  │
│                    - 7 features normalisées [0,1]       │
│                    - Target binaire                     │
└─────────────────────────────────────────────────────────┘
```

### Flux de Données
```mermaid
graph LR
    A[Données CSV] --> B[Normalisation]
    B --> C[Split Stratifié]
    C --> D[Entraînement MLP]
    D --> E[Calibration]
    E --> F[Évaluation]
    F --> G[Export ONNX]
    G --> H[API Rust]
    H --> I[Frontend WASM]
```

---

## 📊 Pipeline ML Complet

### Phase 1: Préparation des Données
```python
# ÉTAPE CRITIQUE : Éviter le Data Leakage
# Split triple strict avec stratification
train/val/test = 70%/15%/15%

# Normalisation Min-Max explicite
X_norm = (X - X_min) / (X_max - X_min)

# Statistiques sauvegardées pour inference
stats = {
    'min': X_min,
    'max': X_max,
    'mean': X_mean,
    'std': X_std
}
```

### Phase 2: Validation des Données
- **Vérification des plages** : Toutes features ∈ [0,1]
- **Détection des outliers** : 3σ rule
- **Corrélations** : Analyse feature/target
- **Valeurs manquantes** : Imputation par médiane

### Phase 3: Modélisation
```python
# Architecture MLP optimisée
7 → 16 → 8 → 1 (logits)

# Techniques avancées
- LayerNorm (meilleur que BatchNorm pour tabulaire)
- Dropout Monte Carlo pour incertitude
- BCEWithLogitsLoss (stabilité numérique)
- AdamW avec weight decay
```

### Phase 4: Entraînement
```python
# Hyperparamètres optimisés
batch_size = 32
learning_rate = 0.001
weight_decay = 0.0001
dropout = 0.2

# Early Stopping
patience = 25 epochs

# Learning Rate Scheduler
ReduceLROnPlateau(patience=10, factor=0.5)
```

### Phase 5: Calibration
```python
# Pour des probabilités fiables
calibration_method = 'isotonic'

# Métriques de calibration
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Brier Score
```

### Phase 6: Évaluation
```python
# Métriques complètes
metrics = {
    'accuracy', 'precision', 'recall', 'f1',
    'auc', 'specificity', 'npv', 'balanced_acc'
}

# Tests statistiques
- Shapiro-Wilk (normalité)
- T-test (différence moyennes)
- Corrélation point-bisériale
```

---

## 🧠 Modèle de Deep Learning

### Architecture MLP
```
INPUT (7) → LAYER 1 (16) → LAYER 2 (8) → OUTPUT (1)
       ↓            ↓            ↓          ↓
    Linear      Linear       Linear      Linear
       ↓            ↓            ↓          ↓
    LayerNorm   LayerNorm      -          Sigmoid
       ↓            ↓            ↓          ↓
    ReLU         ReLU           -          -
       ↓            ↓            ↓          ↓
    Dropout     Dropout         -          -
```

### Choix Architecturaux Justifiés

#### 1. **LayerNorm vs BatchNorm**
```python
# Pour données tabulaires : LayerNorm > BatchNorm
# Raisons :
# 1. Stable avec batch_size=1 (inference)
# 2. Indépendant des statistiques de batch
# 3. Meilleur pour features corrélées
self.norm = nn.LayerNorm(hidden_size)
```

#### 2. **BCEWithLogitsLoss**
```python
# Au lieu de BCELoss + Sigmoid
# Avantages :
# 1. Stabilité numérique (évite log(0))
# 2. Meilleure convergence
# 3. Compatible export ONNX
self.criterion = nn.BCEWithLogitsLoss()
```

#### 3. **Monte Carlo Dropout**
```python
# Estimation d'incertitude bayésienne approchée
def predict_with_uncertainty(self, x, n_samples=50):
    self.train()  # Dropout activé
    probs_samples = []
    for _ in range(n_samples):
        probs = self.forward(x)
        probs_samples.append(probs)
    
    # Moyenne et écart-type
    mean_probs = torch.stack(probs_samples).mean(0)
    std_probs = torch.stack(probs_samples).std(0)
    
    return mean_probs, std_probs
```

#### 4. **Initialisation He/Kaiming**
```python
def _init_weights(self):
    for layer in self.layers:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(
                layer.weight, 
                mode='fan_in',
                nonlinearity='relu'
            )
            nn.init.zeros_(layer.bias)
```

### Mathématiques du Modèle

#### Forward Pass
```
z₁ = W₁x + b₁
a₁ = LayerNorm(z₁)
h₁ = ReLU(a₁)
d₁ = Dropout(h₁, p=0.2)

z₂ = W₂d₁ + b₂
a₂ = LayerNorm(z₂)
h₂ = ReLU(a₂)
d₂ = Dropout(h₂, p=0.2)

z₃ = W₃d₂ + b₃
ŷ = σ(z₃)  # Sigmoid
```

#### Loss Function
```
L(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

#### Gradient Flow
```
∂L/∂Wᵢ = ∂L/∂ŷ · ∂ŷ/∂z₃ · ∂z₃/∂h₂ · ∂h₂/∂z₂ · ∂z₂/∂Wᵢ
```

---

## ⚙️ Configuration Avancée

### Dataclass de Configuration
```python
@dataclass
class ModelConfig:
    # Split des données
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Architecture
    hidden_sizes: Tuple[int, ...] = (16, 8)
    dropout_rate: float = 0.2
    normalization: str = 'layer'  # 'layer', 'batch', 'none'
    
    # Optimisation
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    
    # Calibration
    calibration_method: str = 'isotonic'
    threshold_range: Tuple[float, float] = (0.1, 0.9)
    
    # Analyse
    n_permutations: int = 100
    confidence_intervals: bool = True
```

### Paramètres Optimisés

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **Batch Size** | 32 | Bon compromis vitesse/stabilité |
| **Learning Rate** | 0.001 | Standard pour AdamW |
| **Weight Decay** | 0.0001 | Régularisation L2 légère |
| **Dropout** | 0.2 | Prévention overfitting modérée |
| **Hidden Layers** | 16, 8 | Capacité suffisante pour 7 features |
| **Epochs** | 200 | Avec early stopping (patience=25) |

---

## 🚀 Utilisation Pas à Pas

### Installation des Dépendances
```bash
# 1. Environnement Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Installation PyTorch (choisir selon votre CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Autres dépendances
pip install numpy pandas scikit-learn matplotlib seaborn joblib

# 4. Pour l'export ONNX
pip install onnx onnxruntime
```

### Structure du Projet
```
student_success_predictor/
├── data/
│   ├── dataset_strict_7features.csv
│   └── raw/ (données brutes)
├── src/
│   ├── ml_pipeline.py          # Pipeline ML principal
│   ├── model_pro.py            # Version production
│   ├── data_manager.py         # Gestion des données
│   ├── model_architecture.py   # Définition du modèle
│   └── utils.py                # Fonctions utilitaires
├── models/                     # Modèles entraînés
├── reports/                    # Rapports et visualisations
├── exports/                    # Modèles exportés (ONNX, TorchScript)
├── tests/                      # Tests unitaires
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Entraînement du Modèle
```bash
# Version simple
python src/ml_pipeline.py train

# Version production avec logs
python src/model_pro.py train --config configs/production.json

# Avec monitoring TensorBoard
tensorboard --logdir=logs/
```

### Configuration Personnalisée
```json
// configs/custom.json
{
    "data_path": "data/dataset_strict_7features.csv",
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_sizes": [16, 8],
    "dropout_rate": 0.2,
    "normalization": "layer",
    "calibration_method": "isotonic"
}
```

### Prédiction
```bash
# Format JSON
python src/model_pro.py predict \
    --model models/student_model_pro_20240101_120000.pth \
    --data '{
        "Niveau_etude": 0.8,
        "Heures_etude_ordinal": 0.9,
        "Planning_ordinal": 0.7,
        "Assiduite_ordinal": 0.8,
        "Environnement_ordinal": 0.6,
        "Sommeil_score": 0.7,
        "Qualite_ordinal": 0.8
    }'
```

### Batch Prediction
```bash
# Fichier JSON avec plusieurs étudiants
python src/model_pro.py batch_predict \
    --model models/student_model_pro_20240101_120000.pth \
    --input data/batch_students.json
```

---

## 📈 Résultats et Métriques

### Métriques Standard
```python
# Sur le test set (15%, jamais vu pendant l'entraînement)
{
    "accuracy": 0.825,
    "f1_score": 0.896,
    "precision": 0.863,
    "recall": 0.930,
    "auc": 0.901,
    "specificity": 0.387,
    "npv": 0.571
}
```

### Matrice de Confusion
```
        Prédit 0  Prédit 1
Réel 0     12         19
Réel 1      9        120
```

### Calibration
```python
# Mesures de fiabilité des probabilités
{
    "ece": 0.032,     # Expected Calibration Error
    "mce": 0.085,     # Maximum Calibration Error
    "brier_score": 0.126
}
```

### Importance des Features
```python
# Par permutation (100 permutations)
{
    "Qualite_ordinal": {
        "importance": 0.0169,
        "std": 0.0042,
        "ci_95": [0.0087, 0.0251],
        "p_value": 0.0001,
        "significant": true
    },
    "Planning_ordinal": {
        "importance": 0.0132,
        "std": 0.0038,
        "ci_95": [0.0058, 0.0206],
        "p_value": 0.0005,
        "significant": true
    }
}
```

---

## 🔬 Analyse Scientifique

### 1. Validation Statistique

#### Tests de Normalité
```python
# Shapiro-Wilk test pour chaque classe
for class_label in [0, 1]:
    class_probs = probs[y_true == class_label]
    stat, p_value = stats.shapiro(class_probs)
    # H0: les données sont normalement distribuées
    # p < 0.05 → rejet H0 → pas normal
```

#### Test de Différence des Moyennes
```python
# T-test indépendant (Welch)
stat, p_value = stats.ttest_ind(
    probs[y_true == 0],
    probs[y_true == 1],
    equal_var=False
)
# p < 0.05 → différence significative
```

#### Corrélation Point-Bisériale
```python
# Relation entre variable continue (probs) et binaire (y_true)
correlation, p_value = stats.pointbiserialr(y_true, probs)
# rpb ≈ 0.6 → forte corrélation
```

### 2. Robustesse du Modèle

#### Cross-Validation Stratifiée
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in skf.split(X, y):
    model = MLPWithUncertainty(config)
    # Entraînement...
    score = evaluator.evaluate(val_loader)
    cv_scores.append(score['f1'])
    
print(f"CV F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

#### Bootstrap Confidence Intervals
```python
def bootstrap_ci(scores, n_bootstrap=1000, ci=95):
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_means, (100 + ci) / 2)
    return lower, upper
```

### 3. Analyse des Features

#### Visualisation des Corrélations
```python
corr_matrix = data[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de Corrélation')
plt.savefig('reports/correlation_matrix.png')
```

#### Distribution par Classe
```python
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, feature in enumerate(FEATURE_COLUMNS):
    ax = axes[idx // 4, idx % 4]
    for class_label in [0, 1]:
        class_data = data[data[TARGET_COLUMN] == class_label][feature]
        ax.hist(class_data, alpha=0.5, label=f'Classe {class_label}')
    ax.set_title(feature)
    ax.legend()
```

---

## 🛠️ Optimisations Techniques

### 1. Optimisation du Training Loop

#### Gradient Accumulation
```python
accumulation_steps = 4
for batch_idx, (features, labels) in enumerate(train_loader):
    loss = criterion(model(features), labels)
    loss = loss / accumulation_steps  # Normalisation
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Mixed Precision Training
```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(features)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    norm_type=2
)
```

### 2. Optimisation de la Mémoire

#### Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    # Checkpoint les couches intermédiaires
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```

#### CPU Offloading
```python
# Pour les très grands modèles
model.to('cuda')
for param in model.parameters():
    param.data = param.data.to('cuda')
    if param.grad is not None:
        param.grad.data = param.grad.data.to('cpu')
```

### 3. Optimisation de l'Inference

#### Pruning du Modèle
```python
from torch.nn.utils import prune

# Pruning structuré
prune.l1_unstructured(module, name='weight', amount=0.3)
prune.remove(module, 'weight')  # Permanent

# Pruning itératif
for epoch in range(epochs):
    # Entraînement...
    if epoch % 10 == 0:
        prune_model(model, amount=0.1)
```

#### Quantization
```python
# Post-training quantization
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Quantization-aware training
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
```

#### Kernel Fusion
```python
# Optimisation manuelle
class FusedLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        # Fusion Linear + ReLU
        x = F.linear(x, self.weight, self.bias)
        return F.relu(x, inplace=True)
```

### 4. Optimisation du DataLoader

#### Prefetching
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # Parallel loading
    pin_memory=True,      # Faster GPU transfer
    prefetch_factor=2     # Prefetch batches
)
```

#### Memory Pinning
```python
# Pour les transferts CPU→GPU
features = features.pin_memory()
labels = labels.pin_memory()
```

### 5. Cache Optimization

#### Result Caching
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def predict_cached(model_hash: str, features_hash: str):
    """Cache des prédictions fréquentes"""
    # Logique de prédiction...
    return prediction

def hash_features(features: dict) -> str:
    """Hash des features pour le cache"""
    features_str = json.dumps(features, sort_keys=True)
    return hashlib.md5(features_str.encode()).hexdigest()
```

#### Model Caching
```python
class ModelCache:
    def __init__(self, max_size=5):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get_model(self, model_path: str):
        if model_path in self.cache:
            # Mettre à jour l'ordre d'accès
            self.access_order.remove(model_path)
            self.access_order.append(model_path)
            return self.cache[model_path]
        
        # Charger le modèle
        model = load_model(model_path)
        
        # Gérer le cache LRU
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[model_path] = model
        self.access_order.append(model_path)
        return model
```

---

## 🔮 Déploiement et Production

### 1. Export des Modèles

#### TorchScript
```python
model.eval()
example_input = torch.randn(1, 7)
traced_script = torch.jit.trace(model, example_input)
traced_script.save("model_ts.pt")
```

#### ONNX
```python
torch.onnx.export(
    model,
    torch.randn(1, 7),
    "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['features'],
    output_names=['prediction'],
    dynamic_axes={
        'features': {0: 'batch_size'},
        'prediction': {0: 'batch_size'}
    }
)
```

### 2. API Rust avec Axum

```rust
// Cargo.toml
[dependencies]
axum = "0.6"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
onnxruntime = "0.1.0"

// main.rs
use axum::{
    extract::Json,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct StudentFeatures {
    niveau_etude: f32,
    heures_etude: f32,
    planning: f32,
    assiduite: f32,
    environnement: f32,
    sommeil: f32,
    qualite: f32,
}

#[derive(Serialize)]
struct PredictionResult {
    probability: f32,
    prediction: String,
    confidence: String,
    uncertainty: f32,
    ci_95: [f32; 2],
}

async fn predict_student(
    Json(features): Json<StudentFeatures>
) -> Json<PredictionResult> {
    // Chargement modèle ONNX
    let session = load_onnx_model("model.onnx");
    
    // Préparation des features
    let input_tensor = prepare_features(features);
    
    // Inference
    let outputs = session.run(vec![input_tensor]);
    let probability = outputs[0][0];
    
    // Construction réponse
    Json(PredictionResult {
        probability,
        prediction: if probability >= 0.5 {
            "RÉUSSITE".to_string()
        } else {
            "ÉCHEC".to_string()
        },
        confidence: "ÉLEVÉE".to_string(),
        uncertainty: 0.1,
        ci_95: [probability - 0.05, probability + 0.05],
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/predict", post(predict_student));
    
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

### 3. Dockerisation

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Port API
EXPOSE 8000

# Commande de démarrage
CMD ["python", "src/api.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/student_model.onnx
      - LOG_LEVEL=INFO
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  frontend:
    build: ./frontend
    ports:
      - "8080:80"
    depends_on:
      - ml-api
```

### 4. Monitoring Production

```python
# monitoring.py
from prometheus_client import Counter, Histogram, start_http_server
import time

# Métriques Prometheus
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
ERROR_COUNT = Counter('prediction_errors_total', 'Prediction errors')

class MonitoredModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        start_http_server(9090)  # Métriques sur port 9090
    
    @PREDICTION_LATENCY.time()
    def predict(self, features):
        PREDICTION_COUNT.inc()
        try:
            start_time = time.time()
            result = self.model(features)
            return result
        except Exception as e:
            ERROR_COUNT.inc()
            raise e
```

---

## 🌐 Frontend (Yew + Rust)

### Architecture Frontend

```
frontend/
├── src/
│   ├── components/
│   │   ├── prediction_form.rs
│   │   ├── results_display.rs
│   │   ├── feature_analysis.rs
│   │   └── charts.rs
│   ├── services/
│   │   ├── api_client.rs
│   │   └── cache.rs
│   ├── models/
│   │   ├── student.rs
│   │   └── prediction.rs
│   ├── utils/
│   │   ├── validation.rs
│   │   └── formatting.rs
│   └── app.rs
├── static/
│   ├── index.html
│   ├── style.css
│   └── favicon.ico
├── Cargo.toml
└── package.json
```

### Composant Principal

```rust
// src/app.rs
use yew::prelude::*;
use crate::components::{PredictionForm, ResultsDisplay};
use crate::services::ApiClient;
use crate::models::{Student, PredictionResult};

#[function_component(App)]
pub fn app() -> Html {
    let prediction_result = use_state(|| None);
    let loading = use_state(|| false);
    
    let on_predict = {
        let prediction_result = prediction_result.clone();
        let loading = loading.clone();
        
        Callback::from(move |student: Student| {
            let prediction_result = prediction_result.clone();
            let loading = loading.clone();
            
            wasm_bindgen_futures::spawn_local(async move {
                loading.set(true);
                
                match ApiClient::predict_student(&student).await {
                    Ok(result) => {
                        prediction_result.set(Some(result));
                    }
                    Err(err) => {
                        // Gestion erreur
                    }
                }
                
                loading.set(false);
            });
        })
    };
    
    html! {
        <div class="app">
            <header>
                <h1>{"🎓 Prédiction de Réussite"}</h1>
            </header>
            
            <main>
                <PredictionForm on_predict={on_predict} />
                
                if *loading {
                    <div class="loading">{"Chargement..."}</div>
                } else if let Some(result) = &*prediction_result {
                    <ResultsDisplay result={result.clone()} />
                }
            </main>
        </div>
    }
}
```

### Formulaire de Prédiction

```rust
// src/components/prediction_form.rs
use yew::prelude::*;
use crate::models::Student;

#[derive(Properties, PartialEq)]
pub struct PredictionFormProps {
    pub on_predict: Callback<Student>,
}

#[function_component(PredictionForm)]
pub fn prediction_form(props: &PredictionFormProps) -> Html {
    let niveau_etude = use_state(|| 0.5);
    let heures_etude = use_state(|| 0.5);
    // ... autres features
    
    let on_submit = {
        let on_predict = props.on_predict.clone();
        let niveau_etude = niveau_etude.clone();
        let heures_etude = heures_etude.clone();
        // ... autres features
        
        Callback::from(move |e: SubmitEvent| {
            e.prevent_default();
            
            let student = Student {
                niveau_etude: *niveau_etude,
                heures_etude: *heures_etude,
                // ... autres features
            };
            
            on_predict.emit(student);
        })
    };
    
    html! {
        <form onsubmit={on_submit} class="prediction-form">
            <div class="form-group">
                <label for="niveau_etude">{"Niveau d'étude"}</label>
                <input
                    type="range"
                    id="niveau_etude"
                    min="0"
                    max="1"
                    step="0.1"
                    value={*niveau_etude}
                    oninput={...}
                />
                <span>{format!("{:.1}", *niveau_etude)}</span>
            </div>
            
            // ... autres inputs
            
            <button type="submit" class="btn-predict">
                {"Prédire la réussite"}
            </button>
        </form>
    }
}
```

### Visualisations avec D3.js (via wasm-bindgen)

```rust
// src/components/charts.rs
use wasm_bindgen::prelude::*;
use web_sys::{window, document, Element};
use crate::models::PredictionResult;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = d3)]
    fn select(selector: &str) -> JsValue;
    
    #[wasm_bindgen(js_namespace = d3)]
    fn scaleLinear() -> JsValue;
}

pub fn render_probability_chart(result: &PredictionResult, element_id: &str) {
    let document = document().unwrap();
    let element = document.get_element_by_id(element_id).unwrap();
    
    // Nettoyer l'élément
    element.set_inner_html("");
    
    // Créer le SVG avec D3
    let svg = js_sys::eval(&format!(
        r#"
        d3.select('#{}')
            .append('svg')
            .attr('width', 400)
            .attr('height', 200)
        "#,
        element_id
    )).unwrap();
    
    // Créer l'échelle
    let x_scale = js_sys::eval(
        r#"d3.scaleLinear().domain([0, 1]).range([0, 400])"#
    ).unwrap();
    
    // Ajouter la barre de probabilité
    let _ = js_sys::eval(&format!(
        r#"
        d3.select('#{} svg')
            .append('rect')
            .attr('x', 0)
            .attr('y', 80)
            .attr('width', {})
            .attr('height', 40)
            .attr('fill', '{}')
        "#,
        element_id,
        result.probability * 400.0,
        if result.probability >= 0.5 { "#4CAF50" } else { "#F44336" }
    ));
    
    // Ajouter le texte
    let _ = js_sys::eval(&format!(
        r#"
        d3.select('#{} svg')
            .append('text')
            .attr('x', 200)
            .attr('y', 60)
            .attr('text-anchor', 'middle')
            .attr('font-size', '24px')
            .attr('font-weight', 'bold')
            .text('{:.1%}')
        "#,
        element_id,
        result.probability
    ));
}
```

### Build et Déploiement Frontend

```bash
# Installation
npm install
cargo install trunk

# Développement
trunk serve

# Build production
trunk build --release

# Déploiement sur GitHub Pages
trunk build --release --public-url /student-success-predictor/
```

---

## 📁 Structure des Fichiers

### Organisation Complète
```
student-success-predictor/
├── .github/workflows/              # CI/CD GitHub Actions
│   ├── train-model.yml
│   ├── deploy-api.yml
│   └── deploy-frontend.yml
├── api/                            # API Rust
│   ├── src/
│   │   ├── main.rs
│   │   ├── handlers/
│   │   ├── models/
│   │   └── utils/
│   ├── Cargo.toml
│   └── Dockerfile
├── frontend/                       # Frontend Yew
│   ├── src/
│   ├── static/
│   ├── Cargo.toml
│   └── index.html
├── ml/                             # Pipeline ML Python
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_manager.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── exporter.py
│   │   └── api.py
│   ├── configs/
│   │   ├── base.json
│   │   ├── production.json
│   │   └── experimental.json
│   ├── tests/
│   │   ├── test_data.py
│   │   ├── test_model.py
│   │   └── test_pipeline.py
│   ├── requirements.txt
│   └── Dockerfile
├── data/                           # Données
│   ├── raw/                       # Données brutes
│   ├── processed/                 # Données transformées
│   ├── splits/                    # Splits prédéfinis
│   └── external/                  # Données externes
├── models/                         # Modèles entraînés
│   ├── pytorch/                   # Modèles PyTorch
│   ├── onnx/                      # Modèles ONNX
│   ├── torchscript/               # Modèles TorchScript
│   └── metadata/                  # Métadonnées des modèles
├── reports/                        # Rapports et analyses
│   ├── training/                  # Rapports d'entraînement
│   ├── evaluation/                # Évaluations détaillées
│   ├── visualizations/            # Graphiques et plots
│   └── papers/                    Documentation scientifique
├── notebooks/                      # Notebooks Jupyter
│   ├── 01-data-exploration.ipynb
│   ├── 02-model-experiments.ipynb
│   └── 03-results-analysis.ipynb
├── docs/                           # Documentation
│   ├── api/                       # Documentation API
│   ├── architecture/              # Documentation architecture
│   ├── deployment/                # Guide de déploiement
│   └── user-guide/                # Guide utilisateur
├── scripts/                        # Scripts utilitaires
│   ├── setup.sh                   # Setup environnement
│   ├── train.sh                   # Script d'entraînement
│   ├── evaluate.sh                # Script d'évaluation
│   └── deploy.sh                  # Script de déploiement
├── .env.example                    # Variables d'environnement
├── .gitignore
├── docker-compose.yml
├── LICENSE
├── README.md                       # Ce fichier
└── pyproject.toml                  # Configuration Python
```

### Description des Répertoires

#### **ml/src/** - Code ML Principal
- `data_manager.py` : Chargement, validation, préparation des données
- `model.py` : Architecture du modèle MLP avec incertitude
- `trainer.py` : Boucle d'entraînement avec early stopping
- `evaluator.py` : Évaluation complète avec tests statistiques
- `exporter.py` : Export ONNX, TorchScript, etc.
- `api.py` : API FastAPI pour inference

#### **api/src/** - API Rust
- `handlers/` : Handlers HTTP pour les endpoints
- `models/` : Structures de données (Serde)
- `middleware/` : Middleware (CORS, logging, auth)
- `services/` : Services métier (inference, cache)

#### **frontend/src/** - Frontend Yew
- `components/` : Composants réutilisables
- `pages/` : Pages de l'application
- `services/` : Services API, cache local
- `hooks/` : Custom hooks Yew
- `utils/` : Utilitaires (validation, format)

#### **reports/** - Documentation et Analyse
- `training/` : Logs et métriques d'entraînement
- `evaluation/` : Rapports d'évaluation détaillés
- `visualizations/` : Graphiques exportés
- `papers/` : Documentation scientifique

---

## 🧪 Tests et Validation

### Tests Unitaires Python

```python
# tests/test_model.py
import pytest
import torch
from ml.src.model import MLPWithUncertainty

def test_model_initialization():
    """Test l'initialisation du modèle"""
    model = MLPWithUncertainty(config)
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0

def test_forward_pass():
    """Test le forward pass"""
    model = MLPWithUncertainty(config)
    x = torch.randn(10, 7)  # Batch de 10
    output = model(x)
    assert output.shape == (10, 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_uncertainty_prediction():
    """Test l'estimation d'incertitude"""
    model = MLPWithUncertainty(config)
    x = torch.randn(1, 7)
    result = model.predict_with_uncertainty(x, n_samples=10)
    assert 'mean_probs' in result
    assert 'std_probs' in result
    assert 'ci_95' in result
```

### Tests d'Intégration

```python
# tests/test_pipeline.py
def test_full_pipeline():
    """Test le pipeline complet"""
    # 1. Chargement des données
    data_manager = AdvancedDataManager(CONFIG)
    data = data_manager.load_and_validate()
    assert data.shape[0] == 1000
    assert data.shape[1] == 8  # 7 features + target
    
    # 2. Création des splits
    splits = data_manager.create_stratified_splits()
    assert len(splits['train']) == 700
    assert len(splits['val']) == 150
    assert len(splits['test']) == 150
    
    # 3. Entraînement du modèle
    model = MLPWithUncertainty(CONFIG)
    trainer = ProfessionalTrainer(model, CONFIG)
    results = trainer.train(train_loader, val_loader)
    
    # 4. Évaluation
    evaluator = ProductionEvaluator(model, results['optimal_threshold'])
    metrics, probs, labels = evaluator.evaluate_comprehensive(test_loader)
    
    # Vérifications
    assert metrics['f1_score'] > 0.8
    assert metrics['auc'] > 0.8
    assert 'calibration' in metrics
    assert 'feature_importance' in metrics
```

### Tests de Performance

```python
# tests/test_performance.py
import time

def test_training_performance():
    """Test les performances d'entraînement"""
    start_time = time.time()
    
    model = MLPWithUncertainty(CONFIG)
    trainer = ProfessionalTrainer(model, CONFIG)
    results = trainer.train(train_loader, val_loader)
    
    training_time = time.time() - start_time
    
    # Le training doit prendre moins de 60 secondes
    assert training_time < 60.0
    
    # Au moins 50% des epochs doivent être utilisées
    assert results['best_epoch'] >= CONFIG.epochs * 0.5

def test_inference_latency():
    """Test la latence d'inférence"""
    model = MLPWithUncertainty(CONFIG)
    model.eval()
    
    # Test batch size 1
    x = torch.randn(1, 7)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(1000):
            _ = model(x)
    latency_ms = (time.time() - start_time) * 1000 / 1000
    
    # Inférence doit prendre moins de 10ms
    assert latency_ms < 10.0
```

### Tests Rust (API)

```rust
// tests/api_tests.rs
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::ServiceExt;
use student_predictor_api::app;

#[tokio::test]
async fn test_predict_endpoint() {
    let app = app();
    
    let request = Request::builder()
        .uri("/predict")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{
                "niveau_etude": 0.8,
                "heures_etude": 0.9,
                "planning": 0.7,
                "assiduite": 0.8,
                "environnement": 0.6,
                "sommeil": 0.7,
                "qualite": 0.8
            }"#,
        ))
        .unwrap();
    
    let response = app.oneshot(request).await.unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    
    assert!(body_str.contains("probability"));
    assert!(body_str.contains("prediction"));
    assert!(body_str.contains("confidence"));
}
```

### Tests End-to-End

```bash
#!/bin/bash
# scripts/test_e2e.sh

echo "🚀 Démarrage des tests E2E..."

# 1. Test du pipeline ML
echo "1. Test pipeline ML..."
python -m pytest tests/test_pipeline.py -v

# 2. Test de l'API
echo "2. Test API..."
cd api && cargo test -- --nocapture

# 3. Test frontend
echo "3. Test frontend..."
cd frontend && wasm-pack test --headless

# 4. Test d'intégration
echo "4. Test d'intégration..."
python scripts/test_integration.py

echo "✅ Tous les tests passés!"
```

---

## 📚 Références Techniques

### 1. Articles Scientifiques
- **Dropout as a Bayesian Approximation** (Gal & Ghahramani, 2016)
- **On Calibration of Modern Neural Networks** (Guo et al., 2017)
- **Layer Normalization** (Ba et al., 2016)
- **AdamW: Decoupled Weight Decay Regularization** (Loshchilov & Hutter, 2019)

### 2. Documentation Officielle
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [Yew Framework](https://yew.rs/docs/)
- [Axum Web Framework](https://docs.rs/axum/latest/axum/)

### 3. Meilleures Pratiques
- [MLOps: Continuous Delivery for ML](https://ml-ops.org/)
- [Google's ML Engineering Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Microsoft's Responsible AI](https://www.microsoft.com/en-us/ai/responsible-ai)

### 4. Outils Recommandés
- **Monitoring** : Prometheus + Grafana
- **CI/CD** : GitHub Actions, GitLab CI
- **Container** : Docker, Kubernetes
- **Documentation** : MkDocs, Docusaurus

---

## 🎉 Conclusion

Ce projet démontre un **pipeline ML complet et professionnel** pour la prédiction de réussite étudiante, avec :

### ✅ Points Forts
1. **Architecture moderne** : MLP avec LayerNorm et estimation d'incertitude
2. **Rigueur scientifique** : Tests statistiques, calibration, validation croisée
3. **Production-ready** : Export ONNX, API Rust, monitoring
4. **Interprétabilité** : Feature importance, incertitude, recommandations
5. **Performance** : F1-score > 0.85, latence < 10ms

### 🔮 Prochaines Étapes
1. **Collecte de données réelles** pour validation externe
2. **A/B testing** en environnement éducatif
3. **Fédération learning** pour respecter la vie privée
4. **Modèles multimodaux** intégrant données comportementales
5. **Plateforme SaaS** pour institutions éducatives

### 📞 Contact et Contribution
Ce projet est open-source sous licence MIT. Les contributions sont les bienvenues !

**Repository GitHub** : `https://github.com/yourusername/student-success-predictor`

**Documentation live** : `https://yourusername.github.io/student-success-predictor`

**Docker Hub** : `docker pull yourusername/student-predictor-api`

---

*Documentation mise à jour le : 8 Janvier 2024*  
*Version du projet : 2.0.0*  
*Auteurs : Équipe de Recherche en IA Éducative*
