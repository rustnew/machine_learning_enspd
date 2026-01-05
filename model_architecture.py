"""
MODEL_ARCHITECTURE.PY
Architecture pure du MLP léger (7 → 16 → 8 → 1)
Sans framework, seulement numpy
"""

import numpy as np
from typing import Dict, Tuple, Optional
import json

class LightMLP:
    """
    MLP léger pour prédiction de réussite académique
    Architecture: 7 → 16 → 8 → 1
    """
    
    def __init__(self, input_size: int = 7, random_state: int = 42):
        """
        Initialise le MLP avec l'architecture spécifiée
        
        Args:
            input_size: Nombre de features d'entrée (7)
            random_state: Graine aléatoire pour reproductibilité
        """
        np.random.seed(random_state)
        
        # ============================================================
        # ARCHITECTURE: 7 → 16 → 8 → 1
        # ============================================================
        
        # Couche 1: 7 → 16
        # Initialisation He pour ReLU
        self.layer1_weights = np.random.randn(input_size, 16) * np.sqrt(2.0 / input_size)
        self.layer1_bias = np.zeros(16)
        
        # Couche 2: 16 → 8
        self.layer2_weights = np.random.randn(16, 8) * np.sqrt(2.0 / 16)
        self.layer2_bias = np.zeros(8)
        
        # Couche de sortie: 8 → 1
        self.output_weights = np.random.randn(8, 1) * np.sqrt(2.0 / 8)
        self.output_bias = np.zeros(1)
        
        # ============================================================
        # HYPERPARAMÈTRES DE RÉGULARISATION
        # ============================================================
        self.dropout_rate = 0.1       # Dropout léger
        self.l2_lambda = 0.001        # Régularisation L2
        
        # ============================================================
        # HISTORIQUE ET CACHE
        # ============================================================
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
        
        self.cache = {}  # Pour la rétropropagation
        self.best_weights = None  # Pour early stopping
        self.best_val_loss = float('inf')
        
        # Moments pour Adam optimizer
        self.m = {}
        self.v = {}
        self._init_adam_moments()
        
        print(f"✅ MLP initialisé: {input_size} → 16 → 8 → 1")
        print(f"   Paramètres totaux: {self.count_parameters():,}")
    
    def _init_adam_moments(self):
        """Initialise les moments pour Adam optimizer"""
        param_names = ['layer1_weights', 'layer1_bias', 
                      'layer2_weights', 'layer2_bias',
                      'output_weights', 'output_bias']
        
        for name in param_names:
            param = getattr(self, name)
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
    
    # ============================================================
    # FONCTIONS D'ACTIVATION ET LOSS
    # ============================================================
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Fonction ReLU"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Dérivée de ReLU"""
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Fonction sigmoid pour classification binaire"""
        # Clip pour éviter overflow
        x_clipped = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Binary Cross Entropy Loss"""
        epsilon = 1e-15  # Éviter log(0)
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + 
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        
        return loss
    
    # ============================================================
    # PROPAGATION AVANT
    # ============================================================
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagation avant complète
        
        Args:
            X: Données d'entrée (batch_size, 7)
            training: Mode entraînement (active dropout)
            
        Returns:
            y_pred: Probabilités prédites (batch_size, 1)
        """
        batch_size = X.shape[0]
        
        # --------------------------------------------------------
        # COUCHE 1: Linear + ReLU + Dropout
        # --------------------------------------------------------
        z1 = X.dot(self.layer1_weights) + self.layer1_bias
        a1 = self.relu(z1)
        
        if training and self.dropout_rate > 0:
            dropout_mask1 = (np.random.random(a1.shape) > self.dropout_rate).astype(float)
            a1 *= dropout_mask1 / (1 - self.dropout_rate)  # Scale inverse
            self.cache['dropout_mask1'] = dropout_mask1
        
        # --------------------------------------------------------
        # COUCHE 2: Linear + ReLU + Dropout
        # --------------------------------------------------------
        z2 = a1.dot(self.layer2_weights) + self.layer2_bias
        a2 = self.relu(z2)
        
        if training and self.dropout_rate > 0:
            dropout_mask2 = (np.random.random(a2.shape) > self.dropout_rate).astype(float)
            a2 *= dropout_mask2 / (1 - self.dropout_rate)
            self.cache['dropout_mask2'] = dropout_mask2
        
        # --------------------------------------------------------
        # COUCHE SORTIE: Linear + Sigmoid
        # --------------------------------------------------------
        z3 = a2.dot(self.output_weights) + self.output_bias
        y_pred = self.sigmoid(z3)
        
        # Cache pour backward pass
        if training:
            self.cache.update({
                'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3
            })
        
        return y_pred
    
    # ============================================================
    # RÉTROPROPAGATION
    # ============================================================
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Rétropropagation du gradient avec régularisation L2
        
        Returns:
            gradients: Dictionnaire des gradients pour chaque paramètre
        """
        m = y_true.shape[0]  # Batch size
        
        # --------------------------------------------------------
        # GRADIENT DE LA LOSS BCE
        # --------------------------------------------------------
        dz3 = y_pred - y_true.reshape(-1, 1)  # (batch_size, 1)
        
        # --------------------------------------------------------
        # COUCHE SORTIE: gradients avec L2
        # --------------------------------------------------------
        dw3 = self.cache['a2'].T.dot(dz3) / m + self.l2_lambda * self.output_weights / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # --------------------------------------------------------
        # COUCHE 2: backprop à travers ReLU et dropout
        # --------------------------------------------------------
        da2 = dz3.dot(self.output_weights.T)  # (batch_size, 8)
        
        # Appliquer masque dropout si présent
        if 'dropout_mask2' in self.cache and self.cache['dropout_mask2'] is not None:
            da2 *= self.cache['dropout_mask2'] / (1 - self.dropout_rate)
        
        dz2 = da2 * self.relu_derivative(self.cache['z2'])
        
        # Gradients avec L2
        dw2 = self.cache['a1'].T.dot(dz2) / m + self.l2_lambda * self.layer2_weights / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # --------------------------------------------------------
        # COUCHE 1: backprop à travers ReLU et dropout
        # --------------------------------------------------------
        da1 = dz2.dot(self.layer2_weights.T)  # (batch_size, 16)
        
        if 'dropout_mask1' in self.cache and self.cache['dropout_mask1'] is not None:
            da1 *= self.cache['dropout_mask1'] / (1 - self.dropout_rate)
        
        dz1 = da1 * self.relu_derivative(self.cache['z1'])
        
        # Gradients avec L2
        dw1 = self.cache['X'].T.dot(dz1) / m + self.l2_lambda * self.layer1_weights / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return {
            'dw1': dw1, 'db1': db1,
            'dw2': dw2, 'db2': db2,
            'dw3': dw3, 'db3': db3
        }
    
    # ============================================================
    # OPTIMISATION (ADAM)
    # ============================================================
    
    def update_parameters(self, grads: Dict, learning_rate: float, t: int):
        """
        Mise à jour des paramètres avec Adam optimizer
        
        Args:
            grads: Dictionnaire des gradients
            learning_rate: Taux d'apprentissage
            t: Numéro d'itération (pour correction de biais)
        """
        # Hyperparamètres Adam
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        # Mettre à jour chaque paramètre
        param_mapping = {
            'dw1': 'layer1_weights', 'db1': 'layer1_bias',
            'dw2': 'layer2_weights', 'db2': 'layer2_bias',
            'dw3': 'output_weights', 'db3': 'output_bias'
        }
        
        for grad_name, param_name in param_mapping.items():
            grad = grads[grad_name]
            param = getattr(self, param_name)
            
            # Mettre à jour les moments
            self.m[param_name] = beta1 * self.m[param_name] + (1 - beta1) * grad
            self.v[param_name] = beta2 * self.v[param_name] + (1 - beta2) * (grad ** 2)
            
            # Correction de biais
            m_hat = self.m[param_name] / (1 - beta1 ** t)
            v_hat = self.v[param_name] / (1 - beta2 ** t)
            
            # Mise à jour
            param_update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            setattr(self, param_name, param - param_update)
    
    # ============================================================
    # PRÉDICTION ET MÉTRIQUES
    # ============================================================
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Prédiction de classe binaire
        
        Args:
            X: Données d'entrée
            threshold: Seuil de décision
            
        Returns:
            Classes prédites (0 ou 1)
        """
        y_prob = self.forward(X, training=False)
        return (y_prob >= threshold).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédiction des probabilités
        
        Returns:
            Probabilités de classe 1
        """
        return self.forward(X, training=False).flatten()
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
        """
        Calcule les métriques de performance
        
        Args:
            y_true: Vraies labels
            y_pred: Labels prédits
            y_prob: Probabilités prédites (pour AUC)
            
        Returns:
            Dictionnaire des métriques
        """
        # Matrice de confusion
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # Métriques de base
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calcul d'AUC simplifié (règle du trapèze)
        auc = 0.5
        if y_prob is not None and len(np.unique(y_true)) == 2:
            # Trier par probabilité décroissante
            sorted_indices = np.argsort(y_prob)[::-1]
            y_true_sorted = y_true[sorted_indices]
            
            # Calcul AUC
            tpr = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
            fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)
            auc = np.trapz(tpr, fpr)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': {
                'TP': int(tp), 'TN': int(tn),
                'FP': int(fp), 'FN': int(fn)
            }
        }
    
    # ============================================================
    # ENTRAÎNEMENT AVEC EARLY STOPPING
    # ============================================================
    
    def train_epoch(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                   learning_rate: float, iteration: int) -> float:
        """
        Entraîne sur un seul batch
        
        Returns:
            Loss du batch
        """
        # Forward pass
        y_pred = self.forward(X_batch, training=True)
        
        # Calcul loss
        batch_loss = self.binary_cross_entropy(y_pred, y_batch)
        
        # Backward pass
        grads = self.backward(y_pred, y_batch)
        
        # Mise à jour des paramètres
        self.update_parameters(grads, learning_rate, iteration)
        
        return batch_loss
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
        """
        Évaluation sur un ensemble de données
        
        Returns:
            loss: Loss moyenne
            metrics: Dictionnaire des métriques
        """
        # Prédiction sans dropout
        y_prob = self.forward(X, training=False)
        y_pred = (y_prob >= 0.5).astype(int).flatten()
        
        # Calcul métriques
        loss = self.binary_cross_entropy(y_prob, y)
        metrics = self.compute_metrics(y, y_pred, y_prob.flatten())
        
        return loss, metrics
    
    # ============================================================
    # UTILITAIRES
    # ============================================================
    
    def count_parameters(self) -> int:
        """Compte le nombre total de paramètres"""
        total = 0
        for param_name in ['layer1_weights', 'layer1_bias', 
                          'layer2_weights', 'layer2_bias',
                          'output_weights', 'output_bias']:
            param = getattr(self, param_name)
            total += np.prod(param.shape)
        return total
    
    def get_weights(self) -> Dict:
        """Retourne tous les poids du modèle"""
        return {
            'layer1_weights': self.layer1_weights.copy(),
            'layer1_bias': self.layer1_bias.copy(),
            'layer2_weights': self.layer2_weights.copy(),
            'layer2_bias': self.layer2_bias.copy(),
            'output_weights': self.output_weights.copy(),
            'output_bias': self.output_bias.copy()
        }
    
    def set_weights(self, weights: Dict):
        """Définit les poids du modèle"""
        for key, value in weights.items():
            setattr(self, key, value.copy())
    
    def save_model(self, filepath: str):
        """Sauvegarde le modèle"""
        model_data = {
            'weights': self.get_weights(),
            'history': self.history,
            'architecture': '7_16_8_1',
            'hyperparameters': {
                'dropout_rate': self.dropout_rate,
                'l2_lambda': self.l2_lambda
            }
        }
        np.savez(filepath, **model_data)
        print(f"✅ Modèle sauvegardé: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Charge un modèle sauvegardé"""
        model_data = np.load(filepath, allow_pickle=True)
        
        # Créer une instance
        model = cls(input_size=model_data['weights'].item()['layer1_weights'].shape[0])
        
        # Charger les poids
        model.set_weights(model_data['weights'].item())
        
        # Charger l'historique
        model.history = model_data['history'].item()
        
        print(f"✅ Modèle chargé: {filepath}")
        return model
    
    def get_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: list, n_permutations: int = 10) -> Dict:
        """
        Calcule l'importance des features par permutation
        
        Args:
            X: Données d'entrée
            y: Labels
            feature_names: Noms des features
            n_permutations: Nombre de permutations par feature
            
        Returns:
            Dictionnaire d'importance des features
        """
        # Métrique de référence
        y_pred = self.predict(X)
        baseline_metrics = self.compute_metrics(y, y_pred)
        baseline_score = baseline_metrics['f1_score']
        
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            permutation_scores = []
            
            for _ in range(n_permutations):
                # Permuter la feature
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                
                # Réévaluer
                y_perm_pred = self.predict(X_permuted)
                perm_metrics = self.compute_metrics(y, y_perm_pred)
                
                permutation_scores.append(baseline_score - perm_metrics['f1_score'])
            
            # Score moyen
            importance_scores[feature_name] = np.mean(permutation_scores)
        
        return importance_scores