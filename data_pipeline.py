"""
DATA_PIPELINE.PY
Pipeline complet de préparation des données
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import json

class DataPipeline:
    """
    Pipeline complet pour le preprocessing des données étudiantes
    """
    
    # FEATURES SÉLECTIONNÉES (selon l'analyse)
    SELECTED_FEATURES = [
        'Niveau_etude',              # 🎓 Pertinent
        'Heures_etude_ordinal',      # ⏱️ TRÈS PERTINENT
        'Planning_ordinal',          # 📅 TRÈS PERTINENT
        'Assiduite_ordinal',         # 🏫 CRITIQUE
        'Environnement_ordinal',     # 🌍 Pertinent
        'Sommeil_score',             # 😴 CRITIQUE
        'Qualite_ordinal'            # 👨‍🏫 TRÈS PERTINENT
    ]
    
    # FEATURES À EXCLURE (selon l'analyse)
    EXCLUDED_FEATURES = [
        'Mois_Inscription',          # ❌ Faible lien causal
        'Problemes_salles_ordinal',  # ❌ Peu discriminant
        'Effectif_ordinal',          # ⚠️ Faible à moyen
        'Materiel_ordinal'           # ⚠️ Faible à moyen
    ]
    
    def __init__(self, data_path: str = 'dataset_strict_7features.csv'):
        """
        Initialise le pipeline de données
        
        Args:
            data_path: Chemin vers le fichier de données
        """
        self.data_path = data_path
        self.scaler_params = None
        self.feature_stats = None
        
    def load_and_validate(self) -> pd.DataFrame:
        """
        Charge et valide les données
        
        Returns:
            DataFrame validé
        """
        print("📁 Chargement des données...")
        
        try:
            df = pd.read_csv(self.data_path)
            print(f"   ✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        except FileNotFoundError:
            print(f"   ❌ Fichier {self.data_path} non trouvé")
            print("   ⚠️  Génération de données synthétiques...")
            df = self._generate_synthetic_data()
        
        # Valider les colonnes requises
        required_columns = self.SELECTED_FEATURES + ['Reussite_binaire']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"   ⚠️  Colonnes manquantes: {missing_columns}")
            print("   ⚠️  Tentative de correction...")
            df = self._fix_missing_columns(df, missing_columns)
        
        # Vérifier la distribution des classes
        class_distribution = df['Reussite_binaire'].value_counts(normalize=True)
        print(f"\n📊 Distribution des classes:")
        for class_val, proportion in class_distribution.items():
            count = (df['Reussite_binaire'] == class_val).sum()
            print(f"   • Classe {class_val}: {count} échantillons ({proportion:.1%})")
        
        # Vérifier l'équilibre
        imbalance_ratio = class_distribution.max() / class_distribution.min()
        if imbalance_ratio > 3:
            print(f"   ⚠️  Classes déséquilibrées (ratio: {imbalance_ratio:.1f})")
        
        return df
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Génère des données synthétiques pour le développement
        
        Args:
            n_samples: Nombre d'échantillons à générer
            
        Returns:
            DataFrame synthétique
        """
        np.random.seed(42)
        
        # Générer des features réalistes
        data = {}
        
        # Niveau_etude (0-1, normalisé)
        data['Niveau_etude'] = np.random.beta(2, 2, n_samples)
        
        # Heures d'étude (0-3, ordinal)
        data['Heures_etude_ordinal'] = np.random.choice([0, 1, 2, 3], n_samples, 
                                                         p=[0.1, 0.2, 0.3, 0.4])
        
        # Planning (0-3, ordinal)
        data['Planning_ordinal'] = np.clip(
            data['Heures_etude_ordinal'] + np.random.randint(-1, 2, n_samples), 0, 3
        )
        
        # Assiduité (0-3, ordinal)
        data['Assiduite_ordinal'] = np.random.choice([0, 1, 2, 3], n_samples,
                                                     p=[0.05, 0.1, 0.2, 0.65])
        
        # Environnement (0-2, ordinal)
        data['Environnement_ordinal'] = np.random.choice([0, 1, 2], n_samples,
                                                         p=[0.1, 0.3, 0.6])
        
        # Sommeil (0-4, score)
        data['Sommeil_score'] = np.random.choice([0, 1, 2, 3, 4], n_samples,
                                                 p=[0.1, 0.2, 0.3, 0.25, 0.15])
        
        # Qualité enseignement (0-3, ordinal)
        data['Qualite_ordinal'] = np.random.choice([0, 1, 2, 3], n_samples,
                                                   p=[0.05, 0.25, 0.55, 0.15])
        
        # Générer la cible avec logique réaliste
        success_prob = (
            data['Heures_etude_ordinal'] * 0.15 +
            data['Planning_ordinal'] * 0.12 +
            data['Assiduite_ordinal'] * 0.20 +
            data['Sommeil_score'] * 0.10 +
            data['Qualite_ordinal'] * 0.15 +
            data['Niveau_etude'] * 0.10 +
            data['Environnement_ordinal'] * 0.08 +
            np.random.normal(0, 0.15, n_samples)
        )
        
        data['Reussite_binaire'] = (success_prob > 0.5).astype(int)
        
        df = pd.DataFrame(data)
        print(f"   ✅ Données synthétiques générées: {df.shape}")
        
        return df
    
    def _fix_missing_columns(self, df: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Tente de corriger les colonnes manquantes
        
        Args:
            df: DataFrame original
            missing_columns: Liste des colonnes manquantes
            
        Returns:
            DataFrame corrigé
        """
        for col in missing_columns:
            if col == 'Reussite_binaire':
                # Générer une cible synthétique
                print(f"     → Génération de {col}...")
                df[col] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
            else:
                # Générer des valeurs aléatoires pour les features
                print(f"     → Génération de {col}...")
                if 'ordinal' in col:
                    df[col] = np.random.randint(0, 4, len(df))
                elif 'score' in col:
                    df[col] = np.random.randint(0, 5, len(df))
                else:
                    df[col] = np.random.random(len(df))
        
        return df
    
    def preprocess(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prétraite les données: sélection, normalisation
        
        Args:
            df: DataFrame brut
            fit_scaler: Si True, ajuste le scaler aux données
            
        Returns:
            X: Features prétraitées (n_samples, n_features)
            y: Labels (n_samples,)
        """
        print("\n🔧 Prétraitement des données...")
        
        # 1. Sélection des features
        print("   1. Sélection des 7 features critiques...")
        X = df[self.SELECTED_FEATURES].values.astype(np.float32)
        y = df['Reussite_binaire'].values.astype(np.float32)
        
        print(f"   → Features: {self.SELECTED_FEATURES}")
        print(f"   → Shape: X={X.shape}, y={y.shape}")
        
        # 2. Calcul des statistiques
        self.feature_stats = {
            'means': X.mean(axis=0),
            'stds': X.std(axis=0),
            'mins': X.min(axis=0),
            'maxs': X.max(axis=0),
            'medians': np.median(X, axis=0)
        }
        
        # 3. Normalisation Min-Max (0-1)
        print("   2. Normalisation Min-Max (0-1)...")
        
        if fit_scaler or self.scaler_params is None:
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X_range = X_max - X_min
            
            # Éviter division par zéro
            X_range[X_range == 0] = 1.0
            
            self.scaler_params = {
                'min': X_min.flatten(),
                'max': X_max.flatten(),
                'range': X_range.flatten()
            }
        
        # Appliquer la normalisation
        X_scaled = (X - self.scaler_params['min']) / self.scaler_params['range']
        
        print(f"   → Normalisation appliquée")
        print(f"   → Range des features: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
        
        return X_scaled, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   val_size: float = 0.15, test_size: float = 0.15,
                   random_state: int = 42) -> Tuple:
        """
        Split stratifié des données
        
        Args:
            X: Features
            y: Labels
            val_size: Proportion pour la validation
            test_size: Proportion pour le test
            random_state: Graine aléatoire
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        np.random.seed(random_state)
        
        print(f"\n📊 Split des données...")
        print(f"   Train: {1-val_size-test_size:.0%}, Val: {val_size:.0%}, Test: {test_size:.0%}")
        
        # Indices par classe
        class_0_idx = np.where(y == 0)[0]
        class_1_idx = np.where(y == 1)[0]
        
        # Mélanger
        np.random.shuffle(class_0_idx)
        np.random.shuffle(class_1_idx)
        
        # Calculer les tailles
        n_test_0 = int(len(class_0_idx) * test_size)
        n_test_1 = int(len(class_1_idx) * test_size)
        
        n_val_0 = int(len(class_0_idx) * val_size)
        n_val_1 = int(len(class_1_idx) * val_size)
        
        # Indices de test
        test_idx = np.concatenate([
            class_0_idx[:n_test_0],
            class_1_idx[:n_test_1]
        ])
        
        # Indices de validation
        val_idx = np.concatenate([
            class_0_idx[n_test_0:n_test_0 + n_val_0],
            class_1_idx[n_test_1:n_test_1 + n_val_1]
        ])
        
        # Indices d'entraînement
        train_idx = np.concatenate([
            class_0_idx[n_test_0 + n_val_0:],
            class_1_idx[n_test_1 + n_val_1:]
        ])
        
        # Mélanger les indices
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)
        
        # Créer les splits
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"\n📈 Distribution finale:")
        print(f"   Train:  {X_train.shape[0]:6d} échantillons "
              f"({X_train.shape[0]/len(X):6.1%})")
        print(f"   Val:    {X_val.shape[0]:6d} échantillons "
              f"({X_val.shape[0]/len(X):6.1%})")
        print(f"   Test:   {X_test.shape[0]:6d} échantillons "
              f"({X_test.shape[0]/len(X):6.1%})")
        
        # Vérifier la distribution des classes
        for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            n_class_1 = np.sum(split_y == 1)
            proportion = n_class_1 / len(split_y)
            print(f"   {split_name}: {n_class_1:4d} réussites ({proportion:.1%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_new_data(self, X_new: np.ndarray) -> np.ndarray:
        """
        Normalise de nouvelles données avec les paramètres existants
        
        Args:
            X_new: Nouvelles données (n_samples, n_features)
            
        Returns:
            Données normalisées
        """
        if self.scaler_params is None:
            raise ValueError("Le scaler doit être ajusté d'abord (appeler preprocess avec fit_scaler=True)")
        
        X_scaled = (X_new - self.scaler_params['min']) / self.scaler_params['range']
        return X_scaled
    
    def save_pipeline(self, filepath: str = 'data_pipeline_params.npz'):
        """
        Sauvegarde les paramètres du pipeline
        
        Args:
            filepath: Chemin de sauvegarde
        """
        pipeline_data = {
            'scaler_params': self.scaler_params,
            'feature_stats': self.feature_stats,
            'selected_features': self.SELECTED_FEATURES,
            'excluded_features': self.EXCLUDED_FEATURES
        }
        
        np.savez(filepath, **pipeline_data)
        print(f"✅ Pipeline sauvegardé: {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str = 'data_pipeline_params.npz'):
        """
        Charge un pipeline sauvegardé
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            Instance de DataPipeline avec paramètres chargés
        """
        pipeline_data = np.load(filepath, allow_pickle=True)
        
        # Créer une instance
        pipeline = cls()
        
        # Charger les paramètres
        pipeline.scaler_params = pipeline_data['scaler_params'].item()
        pipeline.feature_stats = pipeline_data['feature_stats'].item()
        
        print(f"✅ Pipeline chargé: {filepath}")
        return pipeline
    
    def analyze_features(self, df: pd.DataFrame) -> Dict:
        """
        Analyse statistique des features
        
        Args:
            df: DataFrame avec les données
            
        Returns:
            Dictionnaire d'analyse
        """
        print("\n🔍 Analyse statistique des features...")
        
        analysis = {}
        
        for feature in self.SELECTED_FEATURES:
            if feature in df.columns:
                feature_data = df[feature]
                
                # Statistiques
                stats = {
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'median': float(feature_data.median()),
                    'skewness': float(feature_data.skew()),
                    'correlation_with_target': float(df[[feature, 'Reussite_binaire']].corr().iloc[0, 1])
                }
                
                # Catégoriser l'importance
                corr_abs = abs(stats['correlation_with_target'])
                if corr_abs > 0.3:
                    importance = '🔥 TRÈS FORTE'
                elif corr_abs > 0.2:
                    importance = '✅ FORTE'
                elif corr_abs > 0.1:
                    importance = '⚠️ MODÉRÉE'
                else:
                    importance = '❌ FAIBLE'
                
                stats['importance'] = importance
                analysis[feature] = stats
                
                # Affichage
                print(f"   • {feature:25s}: corr={stats['correlation_with_target']:+.3f} - {importance}")
        
        return analysis
    
    def create_batches(self, X: np.ndarray, y: np.ndarray, 
                      batch_size: int = 32, shuffle: bool = True) -> list:
        """
        Crée des batches pour l'entraînement
        
        Args:
            X: Features
            y: Labels
            batch_size: Taille des batches
            shuffle: Si True, mélange les données
            
        Returns:
            Liste de tuples (X_batch, y_batch)
        """
        n_samples = X.shape[0]
        
        if shuffle:
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
        else:
            X_shuffled = X
            y_shuffled = y
        
        batches = []
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            if len(X_batch) == batch_size or i + batch_size >= n_samples:
                batches.append((X_batch, y_batch))
        
        return batches