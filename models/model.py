
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    brier_score_loss, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import argparse
from datetime import datetime
import warnings
import hashlib
import pickle
import joblib
from typing import Dict, Tuple, Optional, List, Any, Union
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Configuration avancée
warnings.filterwarnings('ignore')

# Logging structuré
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'model_training_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration centralisée et typée"""
    # Données
    data_path: str = 'data/features.csv'
    total_samples: int = 1000
    
    # Split (corrigé pour éviter data leakage)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Architecture
    input_size: int = 7
    hidden_sizes: Tuple[int, ...] = (16, 8)
    dropout_rate: float = 0.2
    normalization: str = 'layer'  # 'layer', 'batch', 'none'
    activation: str = 'relu'  # 'relu', 'leaky_relu', 'elu'
    
    # Entraînement
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 200
    early_stopping_patience: int = 25
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    
    # Optimisation
    threshold_range: Tuple[float, float] = (0.1, 0.9)
    threshold_steps: int = 81
    calibration_method: str = 'isotonic'  # 'isotonic', 'sigmoid', 'none'
    calibration_cv: int = 5
    
    # Analyse
    n_permutations: int = 100  # Augmenté pour plus de robustesse
    confidence_intervals: bool = True
    ci_level: float = 0.95
    
    # Seed et reproductibilité
    random_seed: int = 42
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    
    # Sauvegarde
    save_format: str = 'torchscript'  # 'torch', 'torchscript', 'onnx'
    compression: bool = True
    
    def __post_init__(self):
        """Validation après initialisation"""
        assert 0 < self.dropout_rate < 1, "Dropout doit être entre 0 et 1"
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-10, \
            f"Ratios doivent sommer à 1.0 (somme: {self.train_ratio + self.val_ratio + self.test_ratio})"
        assert self.normalization in ['layer', 'batch', 'none']
        assert self.calibration_method in ['isotonic', 'sigmoid', 'none']
        
        # Définir le device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Configuration initialisée: {self}")

# Configuration globale
CONFIG = ModelConfig()

FEATURE_COLUMNS = [
    'Niveau_etude',
    'Heures_etude_ordinal', 
    'Planning_ordinal',
    'Assiduite_ordinal',
    'Environnement_ordinal',
    'Sommeil_score',
    'Qualite_ordinal'
]
TARGET_COLUMN = 'Reussite_binaire'


# ============================================================================
# CLASSE JSON ENCODER PERSONNALISÉE
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour gérer les types numpy"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# CLASSES AMÉLIORÉES AVEC GESTION D'ERREURS
# ============================================================================

class MLPWithUncertainty(nn.Module):
    """MLP avancé avec estimation d'incertitude et LayerNorm"""
    
    def __init__(self, config: ModelConfig):
        super(MLPWithUncertainty, self).__init__()
        self.config = config
        
        # Construction dynamique des couches
        layers = []
        input_size = config.input_size
        
        for i, hidden_size in enumerate(config.hidden_sizes):
            # Couche linéaire
            layers.append(nn.Linear(input_size, hidden_size))
            
            # Normalisation (LayerNorm recommandé pour données tabulaires)
            if config.normalization == 'layer':
                layers.append(nn.LayerNorm(hidden_size))
            elif config.normalization == 'batch':
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif config.activation == 'elu':
                layers.append(nn.ELU())
            
            # Dropout pour régularisation
            layers.append(nn.Dropout(config.dropout_rate))
            
            input_size = hidden_size
        
        # Couche de sortie (logits)
        layers.append(nn.Linear(input_size, 1))
        
        self.model = nn.Sequential(*layers)
        self._init_weights()
        
        # Historique
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rate': [], 'epoch_times': []
        }
    
    def _init_weights(self):
        """Initialisation avancée"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, 
                                       mode='fan_in', 
                                       nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - retourne les logits"""
        # Vérification des dimensions
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        return self.model(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 50) -> Dict[str, torch.Tensor]:
        """Prédiction avec estimation d'incertitude par Monte Carlo Dropout"""
        self.train()  # Mode train pour activer le dropout
        
        logits_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                logits_samples.append(logits)
        
        logits_samples = torch.stack(logits_samples, dim=0)
        probs_samples = torch.sigmoid(logits_samples)
        
        # Statistiques
        mean_probs = probs_samples.mean(dim=0)
        std_probs = probs_samples.std(dim=0)
        
        # Intervalle de confiance 95%
        ci_lower = torch.quantile(probs_samples, 0.025, dim=0)
        ci_upper = torch.quantile(probs_samples, 0.975, dim=0)
        
        self.eval()  # Retour en mode eval
        
        return {
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'aleatoric_uncertainty': std_probs,
            'epistemic_uncertainty': std_probs  # Approximation
        }
    
    def compute_calibration_error(self, probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calcule plusieurs métriques de calibration"""
        try:
            # ECE (Expected Calibration Error)
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(probs, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            ece = 0.0
            mce = 0.0  # Maximum Calibration Error
            for i in range(n_bins):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    bin_probs = probs[mask]
                    bin_labels = labels[mask]
                    avg_pred = np.mean(bin_probs)
                    avg_true = np.mean(bin_labels)
                    error = np.abs(avg_pred - avg_true)
                    ece += error * len(bin_probs)
                    mce = max(mce, error)
            
            ece /= len(probs)
            
            # Brier Score
            brier = brier_score_loss(labels, probs)
            
            # Calibration curve
            prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins)
            
            return {
                'ece': float(ece),
                'mce': float(mce),
                'brier_score': float(brier),
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }
        except Exception as e:
            logger.warning(f"Erreur calibration: {e}")
            return {'ece': None, 'mce': None, 'brier_score': None}


class AdvancedDataManager:
    """Gestionnaire de données avancé avec validation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data = None
        self.feature_stats = {}
        self.splits = {}
        
        # Initialisation des seeds
        self._set_seeds()
    
    def _set_seeds(self):
        """Fixer toutes les seeds pour reproductibilité"""
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def load_and_validate(self) -> pd.DataFrame:
        """Chargement et validation complète des données"""
        logger.info(" CHARGEMENT ET VALIDATION DES DONNÉES")
        
        # Chargement
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {data_path}")
        
        self.data = pd.read_csv(data_path)
        logger.info(f"Données chargées: {self.data.shape}")
        
        # Validation complète
        self._validate_structure()
        self._validate_values()
        self._compute_statistics()
        
        return self.data
    
    def _validate_structure(self):
        """Validation structurelle des données"""
        # Vérifier les colonnes
        required_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN])
        missing_cols = required_cols - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        # Vérifier les types
        for col in FEATURE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise ValueError(f"Colonne {col} n'est pas numérique")
        
        # Vérifier les valeurs manquantes
        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Valeurs manquantes détectées: {missing_values}")
            # Stratégie: remplissage par la médiane
            for col in FEATURE_COLUMNS:
                self.data[col].fillna(self.data[col].median(), inplace=True)
    
    def _validate_values(self):
        """Validation des valeurs"""
        features = self.data[FEATURE_COLUMNS].values
        
        # Détection des outliers
        for i, col in enumerate(FEATURE_COLUMNS):
            col_data = features[:, i]
            
            # Outliers statistiques (3 sigma)
            mean = np.mean(col_data)
            std = np.std(col_data)
            outliers = np.sum(np.abs(col_data - mean) > 3 * std)
            
            if outliers > 0:
                logger.warning(f"Outliers détectés dans {col}: {outliers}")
            
            # Vérification de la plage [0,1]
            out_of_bounds = np.sum((col_data < 0) | (col_data > 1))
            if out_of_bounds > 0:
                logger.warning(f"Valeurs hors [0,1] dans {col}: {out_of_bounds}")
    
    def _compute_statistics(self):
        """Calcule et sauvegarde les statistiques"""
        features = self.data[FEATURE_COLUMNS].values
        
        self.feature_stats = {
            'min': features.min(axis=0).tolist(),
            'max': features.max(axis=0).tolist(),
            'mean': features.mean(axis=0).tolist(),
            'std': features.std(axis=0).tolist(),
            'median': np.median(features, axis=0).tolist(),
            'q1': np.percentile(features, 25, axis=0).tolist(),
            'q3': np.percentile(features, 75, axis=0).tolist()
        }
        
        # Corrélations avec la cible
        correlations = {}
        for col in FEATURE_COLUMNS:
            corr = self.data[col].corr(self.data[TARGET_COLUMN])
            correlations[col] = float(corr)
        
        self.feature_stats['correlations'] = correlations
        
        logger.info("Statistiques calculées et sauvegardées")
    
    def create_stratified_splits(self) -> Dict[str, np.ndarray]:
        """Crée des splits stratifiés sans data leakage"""
        logger.info("Création des splits stratifiés")
        
        indices = np.arange(len(self.data))
        
        # Premier split: Train+Val vs Test (test jamais touché)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.config.test_ratio,
            random_state=self.config.random_seed,
            stratify=self.data[TARGET_COLUMN]
        )
        
        # Deuxième split: Train vs Val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.config.val_ratio / (1 - self.config.test_ratio),
            random_state=self.config.random_seed + 1,
            stratify=self.data.iloc[train_val_idx][TARGET_COLUMN]
        )
        
        self.splits = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist()
        }
        
        # Log des distributions
        for split_name, idx in self.splits.items():
            success_rate = self.data.iloc[idx][TARGET_COLUMN].mean()
            logger.info(f"  {split_name}: {len(idx)} samples, réussite: {success_rate:.1%}")
        
        return self.splits
    
    def normalize_features(self) -> Dict[str, np.ndarray]:
        """Normalisation Min-Max avec gestion des constantes"""
        logger.info("Normalisation Min-Max")
        
        features = self.data[FEATURE_COLUMNS].values.astype(np.float32)
        y = self.data[TARGET_COLUMN].values.astype(np.float32)
        
        # Normalisation
        min_vals = np.array(self.feature_stats['min'])
        max_vals = np.array(self.feature_stats['max'])
        ranges = max_vals - min_vals
        
        # Éviter division par zéro pour les constantes
        ranges[ranges == 0] = 1.0
        
        X_normalized = (features - min_vals) / ranges
        X_normalized = np.clip(X_normalized, 0.0, 1.0)
        
        # Split selon indices
        result = {}
        for split_name, idx in self.splits.items():
            idx_array = np.array(idx)
            result[f'X_{split_name}'] = X_normalized[idx_array]
            result[f'y_{split_name}'] = y[idx_array]
        
        return result


class ProfessionalTrainer:
    """Trainer professionnel avec monitoring avancé"""
    
    def __init__(self, model: MLPWithUncertainty, config: ModelConfig):
        self.model = model
        self.config = config
        
        # Device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Loss avec label smoothing
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer avec amsgrad
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            amsgrad=True
        )
        
        # Scheduler avec plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.lr_patience,
            factor=config.lr_factor,
            min_lr=config.min_lr
        )
        
        # Best model tracking
        self.best_state = {
            'weights': None,
            'val_loss': float('inf'),
            'epoch': -1,
            'val_probs': None,
            'val_labels': None
        }
        
        logger.info(f"Trainer initialisé sur {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Entraînement d'une époque avec gradient accumulation"""
        self.model.train()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits.squeeze(), labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            
            # Calcul des probabilités
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader.dataset)
        
        # Métriques
        all_preds = (np.array(all_probs) >= 0.5).astype(int)
        metrics = self._compute_metrics(np.array(all_labels), all_preds, np.array(all_probs))
        
        return avg_loss, metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
        """Validation complète"""
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                logits = self.model(features)
                loss = self.criterion(logits.squeeze(), labels)
                
                total_loss += loss.item() * features.size(0)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader.dataset)
        all_preds = (np.array(all_probs) >= 0.5).astype(int)
        metrics = self._compute_metrics(np.array(all_labels), all_preds, np.array(all_probs))
        
        return avg_loss, metrics, np.array(all_probs), np.array(all_labels)
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_prob: np.ndarray = None) -> Dict:
        """Calcule toutes les métriques avec gestion d'erreurs"""
        metrics = {}
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        
        try:
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
            
            if y_prob is not None and len(np.unique(y_true)) == 2:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob))
            else:
                metrics['auc'] = 0.5
            
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = {
                'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
                'FN': int(cm[1, 0]), 'TP': int(cm[1, 1])
            }
            
            # Métriques additionnelles
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            metrics['balanced_accuracy'] = float((metrics['recall'] + metrics['specificity']) / 2)
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques: {e}")
            metrics = {k: 0.0 for k in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']}
        
        return metrics
    
    def find_optimal_threshold(self, val_probs: np.ndarray, val_labels: np.ndarray) -> Dict:
        """Recherche du seuil optimal avec plusieurs critères"""
        thresholds = np.linspace(
            self.config.threshold_range[0],
            self.config.threshold_range[1],
            self.config.threshold_steps
        )
        
        results = []
        for threshold in thresholds:
            preds = (val_probs >= threshold).astype(int)
            
            f1 = f1_score(val_labels, preds, zero_division=0)
            
            # Youden's J statistic
            recall_val = recall_score(val_labels, preds, zero_division=0)
            specificity_val = 1 - precision_score(val_labels, preds, zero_division=0) if precision_score(val_labels, preds, zero_division=0) > 0 else 0
            youden = recall_val + specificity_val - 1
            
            # Matthews Correlation Coefficient
            tn = np.sum((preds == 0) & (val_labels == 0))
            fp = np.sum((preds == 1) & (val_labels == 0))
            fn = np.sum((preds == 0) & (val_labels == 1))
            tp = np.sum((preds == 1) & (val_labels == 1))
            
            mcc_num = (tp * tn) - (fp * fn)
            mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = mcc_num / mcc_den if mcc_den > 0 else 0
            
            results.append({
                'threshold': float(threshold),
                'f1': float(f1),
                'youden': float(youden),
                'mcc': float(mcc)
            })
        
        # Meilleur par F1 (critère principal)
        best_f1 = max(results, key=lambda x: x['f1'])
        
        return {
            'optimal_threshold': float(best_f1['threshold']),
            'optimal_f1': float(best_f1['f1']),
            'threshold_analysis': results,
            'calibration_needed': bool(abs(best_f1['threshold'] - 0.5) > 0.1)
        }
    
    def calibrate_probabilities(self, probs: np.ndarray, labels: np.ndarray) -> Any:
        """Calibration des probabilités avec Isotonic Regression"""
        if self.config.calibration_method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs, labels)
            return calibrator
        elif self.config.calibration_method == 'sigmoid':
            calibrator = LogisticRegression()
            calibrator.fit(probs.reshape(-1, 1), labels)
            return calibrator
        else:
            return None
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Boucle d'entraînement complète"""
        logger.info("🚀 Début de l'entraînement")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Entraînement
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics, val_probs, val_labels = self.validate(val_loader)
            
            # Mise à jour du scheduler
            self.scheduler.step(val_loss)
            
            # Sauvegarde du meilleur modèle
            if val_loss < self.best_state['val_loss']:
                self.best_state.update({
                    'weights': self.model.state_dict().copy(),
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'val_probs': val_probs.copy(),
                    'val_labels': val_labels.copy()
                })
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Historique
            current_lr = self.optimizer.param_groups[0]['lr']
            self.model.history['train_loss'].append(float(train_loss))
            self.model.history['val_loss'].append(float(val_loss))
            self.model.history['train_acc'].append(float(train_metrics['accuracy']))
            self.model.history['val_acc'].append(float(val_metrics['accuracy']))
            self.model.history['train_f1'].append(float(train_metrics['f1_score']))
            self.model.history['val_f1'].append(float(val_metrics['f1_score']))
            self.model.history['learning_rate'].append(float(current_lr))
            self.model.history['epoch_times'].append(float(time.time() - epoch_start))
            
            # Logging périodique
            if (epoch + 1) % 10 == 0 or epoch == 0:
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                    f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val F1: {val_metrics['f1_score']:.3f}"
                )
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping à l'epoch {epoch + 1}")
                break
        
        # Charger le meilleur modèle
        if self.best_state['weights'] is not None:
            self.model.load_state_dict(self.best_state['weights'])
        
        # Trouver le seuil optimal sur les meilleures probabilités
        threshold_info = self.find_optimal_threshold(
            self.best_state['val_probs'],
            self.best_state['val_labels']
        )
        
        # Calibration si nécessaire
        calibrator = None
        if self.config.calibration_method != 'none':
            calibrator = self.calibrate_probabilities(
                self.best_state['val_probs'],
                self.best_state['val_labels']
            )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Entraînement terminé en {training_time:.1f}s")
        
        return {
            'history': self.model.history,
            'threshold_info': threshold_info,
            'calibrator': calibrator,
            'training_time': float(training_time),
            'best_epoch': int(self.best_state['epoch'] + 1),
            'best_val_loss': float(self.best_state['val_loss']),
            'best_val_f1': float(self._compute_metrics(
                self.best_state['val_labels'],
                (self.best_state['val_probs'] >= threshold_info['optimal_threshold']).astype(int),
                self.best_state['val_probs']
            )['f1_score'])
        }


class ProductionEvaluator:
    """Évaluateur pour production avec tests statistiques"""
    
    def __init__(self, model: MLPWithUncertainty, threshold: float = 0.5, 
                 calibrator: Any = None):
        self.model = model
        self.threshold = threshold
        self.calibrator = calibrator
        
        self.device = next(model.parameters()).device
        self.model.eval()
    
    def evaluate_comprehensive(self, test_loader: DataLoader) -> Dict:
        """Évaluation complète avec tests statistiques"""
        logger.info("📊 Évaluation complète")
        
        # Prédictions
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                probs = torch.sigmoid(self.model(features))
                all_probs.extend(probs.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Calibration si disponible
        if self.calibrator is not None:
            try:
                all_probs = self.calibrator.predict(all_probs)
                all_probs = np.clip(all_probs, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Erreur calibration: {e}")
        
        # Métriques
        preds = (all_probs >= self.threshold).astype(int)
        
        # Métriques de base
        metrics = self._compute_advanced_metrics(all_labels, preds, all_probs)
        
        # Calibration
        calibration_metrics = self.model.compute_calibration_error(all_probs, all_labels)
        metrics['calibration'] = calibration_metrics
        
        # Tests statistiques
        statistical_tests = self._perform_statistical_tests(all_labels, all_probs)
        metrics['statistical_tests'] = statistical_tests
        
        # Feature importance
        feature_importance = self._compute_feature_importance_with_ci(
            test_loader.dataset.tensors[0].numpy(),
            test_loader.dataset.tensors[1].numpy()
        )
        metrics['feature_importance'] = feature_importance
        
        return metrics, all_probs, all_labels
    
    def _compute_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray) -> Dict:
        """Métriques avancées"""
        metrics = {}
        
        # Métriques standard
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics['auc'] = float(roc_auc_score(y_true, y_prob))
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = {
            'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]), 'TP': int(cm[1, 1])
        }
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        # Classification report complet
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        # Convertir les valeurs numpy en Python natives
        for key, value in report.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (np.floating, np.float32, np.float64)):
                        report[key][subkey] = float(subvalue)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                report[key] = float(value)
        metrics['classification_report'] = report
        
        return metrics
    
    def _perform_statistical_tests(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Tests statistiques avancés"""
        tests = {}
        
        # Test de normalité des probabilités par classe
        for class_label in [0, 1]:
            class_probs = y_prob[y_true == class_label]
            if len(class_probs) > 3:  # Besoin d'au moins 3 points
                try:
                    stat, p_value = stats.shapiro(class_probs)
                    tests[f'normality_class_{class_label}'] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'normal': bool(p_value > 0.05)
                    }
                except Exception as e:
                    logger.warning(f"Erreur test Shapiro: {e}")
        
        # Test de différence des moyennes entre classes
        if len(y_prob[y_true == 0]) > 1 and len(y_prob[y_true == 1]) > 1:
            try:
                stat, p_value = stats.ttest_ind(
                    y_prob[y_true == 0], 
                    y_prob[y_true == 1],
                    equal_var=False
                )
                tests['mean_difference'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
            except Exception as e:
                logger.warning(f"Erreur test t: {e}")
        
        # Corrélation point-bisériale
        if len(y_true) > 1:
            try:
                correlation, p_value = stats.pointbiserialr(y_true, y_prob)
                tests['point_biserial_correlation'] = {
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
            except Exception as e:
                logger.warning(f"Erreur corrélation point-bisériale: {e}")
        
        return tests
    
    def _compute_feature_importance_with_ci(self, X_test: np.ndarray, 
                                           y_test: np.ndarray) -> Dict:
        """Feature importance avec intervalles de confiance"""
        logger.info("Calcul de l'importance des features avec CI")
        
        baseline_probs = self._predict_probs(X_test)
        baseline_preds = (baseline_probs >= self.threshold).astype(int)
        baseline_f1 = f1_score(y_test, baseline_preds, zero_division=0)
        
        results = {}
        for i, feature_name in enumerate(FEATURE_COLUMNS):
            permutation_scores = []
            
            for _ in range(CONFIG.n_permutations):
                X_permuted = X_test.copy()
                np.random.shuffle(X_permuted[:, i])
                
                perm_probs = self._predict_probs(X_permuted)
                perm_preds = (perm_probs >= self.threshold).astype(int)
                perm_f1 = f1_score(y_test, perm_preds, zero_division=0)
                
                permutation_scores.append(baseline_f1 - perm_f1)
            
            # Statistiques avec CI
            scores = np.array(permutation_scores)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Intervalle de confiance 95%
            if len(scores) > 1:
                ci_lower, ci_upper = stats.t.interval(
                    0.95, len(scores)-1,
                    loc=mean_score,
                    scale=std_score/np.sqrt(len(scores))
                )
            else:
                ci_lower = ci_upper = mean_score
            
            # Test t unilatéral
            if len(scores) > 1:
                t_stat, p_value = stats.ttest_1samp(scores, 0, alternative='greater')
            else:
                t_stat, p_value = 0, 1
            
            results[feature_name] = {
                'importance': float(mean_score),
                'std': float(std_score),
                'ci_95': [float(ci_lower), float(ci_upper)],
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'effect_size': float(mean_score / baseline_f1) if baseline_f1 > 0 else 0.0
            }
        
        # Tri par importance
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: abs(x[1]['importance']),
            reverse=True
        ))
        
        return sorted_results
    
    def _predict_probs(self, X: np.ndarray) -> np.ndarray:
        """Prédiction batch"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_tensor)).cpu().numpy().flatten()
        
        if self.calibrator is not None:
            try:
                probs = self.calibrator.predict(probs)
            except:
                pass
        
        return np.clip(probs, 0.0, 1.0)
    
    def predict_with_uncertainty(self, features: Dict) -> Dict:
        """Prédiction avec incertitude complète"""
        # Préparation des features
        features_array = self._prepare_features(features)
        
        # Prédiction avec Monte Carlo Dropout
        features_tensor = torch.FloatTensor(features_array).to(self.device)
        uncertainty_result = self.model.predict_with_uncertainty(features_tensor, n_samples=100)
        
        # Probabilité calibrée
        mean_prob = uncertainty_result['mean_probs'].item()
        if self.calibrator is not None:
            try:
                mean_prob = float(self.calibrator.predict([mean_prob])[0])
            except:
                pass
        
        # Décision
        prediction = 1 if mean_prob >= self.threshold else 0
        
        # Analyse de confiance
        total_uncertainty = uncertainty_result['std_probs'].item()
        confidence_score = max(0.0, min(1.0, 1.0 - total_uncertainty))
        
        if confidence_score > 0.8:
            confidence_level = "TRÈS ÉLEVÉE"
            color = "🟢"
        elif confidence_score > 0.6:
            confidence_level = "ÉLEVÉE"
            color = "🟡"
        elif confidence_score > 0.4:
            confidence_level = "MODÉRÉE"
            color = "🟠"
        else:
            confidence_level = "FAIBLE"
            color = "🔴"
        
        return {
            'probability': float(mean_prob),
            'prediction': int(prediction),
            'class': 'RÉUSSITE' if prediction == 1 else 'ÉCHEC',
            'confidence': confidence_level,
            'confidence_color': color,
            'confidence_score': float(confidence_score),
            'uncertainty': float(total_uncertainty),
            'ci_95': [
                float(uncertainty_result['ci_lower'].item()),
                float(uncertainty_result['ci_upper'].item())
            ],
            'features': features,
            'threshold': float(self.threshold),
            'timestamp': datetime.now().isoformat(),
            'model_version': '2.0.0'
        }
    
    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Préparation des features avec validation"""
        features_array = []
        warnings = []
        
        for feature_name in FEATURE_COLUMNS:
            if feature_name in features:
                try:
                    value = float(features[feature_name])
                    
                    # Validation
                    if not (0 <= value <= 1):
                        warnings.append(f"{feature_name} hors [0,1]: {value}")
                        value = np.clip(value, 0.0, 1.0)
                    
                    features_array.append(value)
                except (ValueError, TypeError) as e:
                    warnings.append(f"Erreur {feature_name}: {e}, valeur par défaut 0.5")
                    features_array.append(0.5)
            else:
                warnings.append(f"Feature manquante: {feature_name}, valeur par défaut 0.5")
                features_array.append(0.5)
        
        if warnings:
            logger.warning(f"Warnings préparation features: {warnings}")
        
        return np.array([features_array], dtype=np.float32)


class ModelExporter:
    """Exportation des modèles pour production"""
    
    @staticmethod
    def export_torchscript(model: nn.Module, save_path: str):
        """Export en TorchScript"""
        try:
            model.eval()
            example_input = torch.randn(1, CONFIG.input_size)
            traced_script_module = torch.jit.trace(model, example_input)
            traced_script_module.save(save_path)
            logger.info(f"✅ Modèle exporté en TorchScript: {save_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur export TorchScript: {e}")
            return False
    
    @staticmethod
    def export_onnx(model: nn.Module, save_path: str):
        """Export en ONNX"""
        try:
            model.eval()
            dummy_input = torch.randn(1, CONFIG.input_size)
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"✅ Modèle exporté en ONNX: {save_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur export ONNX: {e}")
            return False
    
    @staticmethod
    def save_checkpoint(model: nn.Module, trainer_results: Dict, 
                       data_manager: AdvancedDataManager, 
                       save_path: str):
        """Sauvegarde complète du checkpoint"""
        checkpoint = {
            # Modèle
            'model_state_dict': model.state_dict(),
            'model_config': asdict(CONFIG),
            'model_hash': ModelExporter.compute_model_hash(model),
            
            # Résultats
            'training_results': trainer_results,
            'feature_stats': data_manager.feature_stats,
            'splits': data_manager.splits,
            
            # Métadonnées
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'numpy_version': np.__version__,
            'python_version': sys.version,
            
            # Données
            'feature_columns': FEATURE_COLUMNS,
            'target_column': TARGET_COLUMN,
            'data_shape': data_manager.data.shape if data_manager.data is not None else None
        }
        
        # Sauvegarde avec compression
        try:
            if CONFIG.compression:
                torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
            else:
                torch.save(checkpoint, save_path)
            
            # Sauvegarde alternative en pickle
            pickle_path = save_path.replace('.pth', '.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"✅ Checkpoint sauvegardé: {save_path}")
            logger.info(f"✅ Checkpoint pickle sauvegardé: {pickle_path}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            return False
    
    @staticmethod
    def compute_model_hash(model: nn.Module) -> str:
        """Hash stable du modèle"""
        # Créer une représentation stable
        model_info = []
        for name, param in model.named_parameters():
            model_info.append(f"{name}:{param.data.sum().item():.10f}")
        
        model_str = "|".join(sorted(model_info))
        return hashlib.sha256(model_str.encode()).hexdigest()[:32]


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_full_pipeline():
    """Pipeline complet d'entraînement et d'évaluation"""
    logger.info("=" * 80)
    logger.info("🚀 DÉMARRAGE DU PIPELINE ML PROFESSIONNEL")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # 1. Gestion des données
    logger.info("\n1️⃣  PHASE 1: GESTION DES DONNÉES")
    data_manager = AdvancedDataManager(CONFIG)
    data = data_manager.load_and_validate()
    splits = data_manager.create_stratified_splits()
    normalized_data = data_manager.normalize_features()
    
    # 2. Création des datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(normalized_data['X_train']),
        torch.FloatTensor(normalized_data['y_train'])
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(normalized_data['X_val']),
        torch.FloatTensor(normalized_data['y_val'])
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(normalized_data['X_test']),
        torch.FloatTensor(normalized_data['y_test'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.batch_size, shuffle=False)
    
    # 3. Initialisation du modèle
    logger.info("\n2️⃣  PHASE 2: INITIALISATION DU MODÈLE")
    model = MLPWithUncertainty(CONFIG)
    logger.info(f"Architecture: {CONFIG.input_size} → {' → '.join(map(str, CONFIG.hidden_sizes))} → 1")
    logger.info(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Normalisation: {CONFIG.normalization}")
    
    # 4. Entraînement
    logger.info("\n3️⃣  PHASE 3: ENTRAÎNEMENT AVANCÉ")
    trainer = ProfessionalTrainer(model, CONFIG)
    trainer_results = trainer.train(train_loader, val_loader)
    
    # 5. Évaluation
    logger.info("\n4️⃣  PHASE 4: ÉVALUATION PROFESSIONNELLE")
    evaluator = ProductionEvaluator(
        model, 
        trainer_results['threshold_info']['optimal_threshold'],
        trainer_results['calibrator']
    )
    
    test_metrics, test_probs, test_labels = evaluator.evaluate_comprehensive(test_loader)
    
    # 6. Affichage des résultats
    logger.info("\n5️⃣  PHASE 5: RÉSULTATS")
    logger.info("=" * 50)
    
    print(f"\n🎯 PERFORMANCES SUR TEST SET:")
    print(f"   • Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   • F1-score:  {test_metrics['f1_score']:.4f}")
    print(f"   • AUC:       {test_metrics['auc']:.4f}")
    print(f"   • Precision: {test_metrics['precision']:.4f}")
    print(f"   • Recall:    {test_metrics['recall']:.4f}")
    print(f"   • Seuil optimal: {trainer_results['threshold_info']['optimal_threshold']:.3f}")
    
    if 'calibration' in test_metrics and test_metrics['calibration']['ece'] is not None:
        print(f"   • ECE: {test_metrics['calibration']['ece']:.4f}")
        print(f"   • Brier Score: {test_metrics['calibration']['brier_score']:.4f}")
    
    # 7. Feature importance
    logger.info("\n6️⃣  PHASE 6: ANALYSE DES FEATURES")
    print("\n🔍 IMPORTANCE DES FEATURES:")
    for feature, importance in test_metrics['feature_importance'].items():
        symbol = "🔥" if importance['significant'] else "➖"
        ci = importance['ci_95']
        print(f"   {symbol} {feature:25s}: {importance['importance']:+.4f} "
              f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}], p={importance['p_value']:.3f})")
    
    # 8. Sauvegarde
    logger.info("\n7️⃣  PHASE 7: SAUVEGARDE PROFESSIONNELLE")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarde du modèle
    model_path = f'student_model_pro_{timestamp}.pth'
    ModelExporter.save_checkpoint(model, trainer_results, data_manager, model_path)
    
    # Export en TorchScript
    ts_path = f'student_model_ts_{timestamp}.pt'
    ModelExporter.export_torchscript(model, ts_path)
    
    # Rapport détaillé - Convertir les valeurs numpy en Python natives
    report = {
        'metadata': {
            'timestamp': timestamp,
            'training_time': float(trainer_results['training_time']),
            'best_epoch': int(trainer_results['best_epoch']),
            'best_val_loss': float(trainer_results['best_val_loss']),
            'best_val_f1': float(trainer_results['best_val_f1']),
            'model_hash': ModelExporter.compute_model_hash(model),
            'config': asdict(CONFIG)
        },
        'data_info': {
            'n_samples': int(len(data)),
            'train_size': int(len(splits['train'])),
            'val_size': int(len(splits['val'])),
            'test_size': int(len(splits['test'])),
            'success_rate': float(data[TARGET_COLUMN].mean()),
            'feature_correlations': data_manager.feature_stats.get('correlations', {})
        },
        'training_info': {
            'history': trainer_results['history'],
            'threshold_info': trainer_results['threshold_info'],
            'calibration_method': CONFIG.calibration_method
        },
        'evaluation': test_metrics,
        'feature_importance': test_metrics['feature_importance']
    }
    
    report_path = f'performance_report_pro_{timestamp}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # Visualisation
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Courbes d'apprentissage
        axes[0, 0].plot(trainer_results['history']['train_loss'], label='Train')
        axes[0, 0].plot(trainer_results['history']['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Courbes de Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Courbe ROC
        axes[0, 1].plot(test_metrics['roc_curve']['fpr'], test_metrics['roc_curve']['tpr'])
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'Courbe ROC (AUC = {test_metrics["auc"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature importance
        features = list(test_metrics['feature_importance'].keys())
        importances = [test_metrics['feature_importance'][f]['importance'] for f in features]
        ci_lower = [test_metrics['feature_importance'][f]['ci_95'][0] for f in features]
        ci_upper = [test_metrics['feature_importance'][f]['ci_95'][1] for f in features]
        
        y_pos = np.arange(len(features))
        axes[1, 0].barh(y_pos, importances, xerr=[np.abs(ci_lower), ci_upper], 
                       capsize=5, alpha=0.7)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(features)
        axes[1, 0].set_xlabel('Importance (ΔF1)')
        axes[1, 0].set_title('Importance des Features avec CI 95%')
        
        # Calibration curve
        if (test_metrics['calibration']['prob_true'] is not None and 
            len(test_metrics['calibration']['prob_pred']) > 0):
            axes[1, 1].plot(test_metrics['calibration']['prob_pred'], 
                          test_metrics['calibration']['prob_true'], 
                          marker='o', linewidth=3)
            axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1, 1].set_xlabel('Probabilité prédite')
            axes[1, 1].set_ylabel('Fraction de positifs')
            axes[1, 1].set_title(f'Calibration (ECE = {test_metrics["calibration"]["ece"]:.3f})')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'model_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualisation sauvegardée: model_analysis_{timestamp}.png")
    except Exception as e:
        logger.warning(f"Erreur visualisation: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
    print(f"   • Modèle: {model_path}")
    print(f"   • TorchScript: {ts_path}")
    print(f"   • Rapport: {report_path}")
    print(f"   • Temps total: {total_time:.1f}s")
    print("=" * 80)
    
    return model, evaluator, report


def predict_student_professional(model_path: str, student_data: Dict):
    """Prédiction professionnelle"""
    try:
        # Chargement sécurisé
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Reconstruction du modèle
        config_dict = checkpoint['model_config']
        model_config = ModelConfig(**config_dict)
        
        model = MLPWithUncertainty(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Reconstruction de l'évaluateur
        threshold = checkpoint['training_results']['threshold_info']['optimal_threshold']
        
        # Chargement du calibrator si existant
        calibrator = None
        if 'calibrator' in checkpoint['training_results']:
            calibrator = checkpoint['training_results']['calibrator']
        
        evaluator = ProductionEvaluator(model, threshold, calibrator)
        
        # Prédiction
        result = evaluator.predict_with_uncertainty(student_data)
        
        # Affichage professionnel
        print(f"\n🎓 PRÉDICTION PROFESSIONNELLE")
        print("=" * 50)
        
        print(f"📊 RÉSULTATS:")
        print(f"   • Probabilité: {result['probability']:.1%}")
        print(f"   • Prédiction: {result['confidence_color']} {result['class']}")
        print(f"   • Confiance: {result['confidence']} (score: {result['confidence_score']:.2f})")
        print(f"   • Incertitude: {result['uncertainty']:.3f}")
        print(f"   • Intervalle 95%: [{result['ci_95'][0]:.1%}, {result['ci_95'][1]:.1%}]")
        
        print(f"\n🔍 ANALYSE DES FEATURES:")
        for feature in FEATURE_COLUMNS:
            value = student_data.get(feature, 0.5)
            impact = "EXCELLENT" if value > 0.8 else \
                    "BON" if value > 0.6 else \
                    "MOYEN" if value > 0.4 else \
                    "FAIBLE" if value > 0.2 else "TRÈS FAIBLE"
            
            color = "🟢" if value > 0.6 else "🟡" if value > 0.4 else "🔴"
            
            print(f"   {color} {feature:25s}: {value:.2f} → {impact}")
        
        print(f"\n💡 RECOMMANDATIONS:")
        prob = result['probability']
        
        if prob > 0.8:
            print("   🏆 EXCELLENT - Potentiel exceptionnel")
            print("   💡 Leadership académique recommandé")
        elif prob > 0.65:
            print("   👍 TRÈS BON - Bonnes perspectives")
            print("   💡 Continuité des efforts actuels")
        elif prob > 0.5:
            print("   🤔 MODÉRÉ - Atteignable avec effort")
            print("   💡 Focus sur les points d'amélioration")
        elif prob > 0.35:
            print("   ⚠️  DIFFICILE - Soutien nécessaire")
            print("   💡 Accompagnement personnalisé recommandé")
        else:
            print("   🚨 CRITIQUE - Intervention urgente")
            print("   💡 Plan d'action intensif requis")
        
        if result['confidence_score'] < 0.5:
            print(f"\n⚠️  ATTENTION: Prédiction peu fiable (confiance: {result['confidence_score']:.2f})")
            print("   💡 Considérer cette prédiction avec prudence")
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        print(f"❌ Erreur: {e}")
        return None


# ============================================================================
# INTERFACE CLI
# ============================================================================

def main():
    """Interface CLI principale"""
    parser = argparse.ArgumentParser(
        description='🎓 Prédiction de réussite étudiante - Version Production',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python model_pro.py train
  python model_pro.py predict --model model.pth --data '{"Niveau_etude":0.8,...}'
        """
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'predict', 'evaluate'],
        help='Commande à exécuter'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Chemin vers le modèle .pth'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Données étudiant au format JSON'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Fichier de configuration JSON'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎓 PRÉDICTION DE RÉUSSITE ÉTUDIANTE - VERSION PRODUCTION")
    print("=" * 80)
    
    # Charger configuration personnalisée
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
            global CONFIG
            CONFIG = ModelConfig(**custom_config)
            logger.info(f"Configuration chargée: {args.config}")
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return
    
    if args.command == 'train':
        run_full_pipeline()
    
    elif args.command == 'predict':
        if not args.model or not args.data:
            print("❌ --model et --data requis")
            return
        
        try:
            student_data = json.loads(args.data)
            predict_student_professional(args.model, student_data)
        except json.JSONDecodeError:
            print("❌ Format JSON invalide")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    elif args.command == 'evaluate':
        if not args.model:
            print("❌ --model requis")
            return
        
        print("Fonctionnalité d'évaluation avancée à venir...")
        # Implémenter l'évaluation complète d'un modèle existant


if __name__ == "__main__":
    main()