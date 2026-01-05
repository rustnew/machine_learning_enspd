"""
MAIN_TRAINING.PY
Script principal d'entraînement et d'évaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from typing import Dict, Tuple

# Import de nos modules
from model_architecture import LightMLP
from data_pipeline import DataPipeline

class TrainingPipeline:
    """
    Pipeline complet d'entraînement et d'évaluation
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise la pipeline d'entraînement
        
        Args:
            config: Configuration d'entraînement
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.data_pipeline = None
        self.training_history = {}
        
    def _get_default_config(self) -> Dict:
        """Configuration par défaut optimisée"""
        return {
            # Architecture
            'input_size': 7,
            'hidden_size1': 16,
            'hidden_size2': 8,
            'output_size': 1,
            
            # Entraînement
            'epochs': 200,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 15,
            
            # Régularisation
            'dropout_rate': 0.1,
            'l2_lambda': 0.001,
            
            # Données
            'val_size': 0.15,
            'test_size': 0.15,
            'random_state': 42,
            
            # Métriques
            'threshold': 0.5,
            'metrics': ['accuracy', 'f1_score', 'auc', 'precision', 'recall']
        }
    
    def run(self, data_path: str = 'dataset_strict_7features.csv'):
        """
        Exécute la pipeline complète
        
        Args:
            data_path: Chemin vers les données
        """
        print("=" * 70)
        print("🎯 PIPELINE D'ENTRAÎNEMENT MLP POUR PRÉDICTION DE RÉUSSITE")
        print("=" * 70)
        
        start_time = time.time()
        
        # ------------------------------------------------------------
        # 1. PRÉPARATION DES DONNÉES
        # ------------------------------------------------------------
        print("\n📊 PHASE 1: PRÉPARATION DES DONNÉES")
        print("-" * 50)
        
        self.data_pipeline = DataPipeline(data_path)
        
        # Chargement
        df = self.data_pipeline.load_and_validate()
        
        # Analyse des features
        feature_analysis = self.data_pipeline.analyze_features(df)
        
        # Préprocessing
        X, y = self.data_pipeline.preprocess(df, fit_scaler=True)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_pipeline.split_data(
            X, y, 
            val_size=self.config['val_size'],
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # ------------------------------------------------------------
        # 2. INITIALISATION DU MODÈLE
        # ------------------------------------------------------------
        print("\n🧠 PHASE 2: INITIALISATION DU MODÈLE")
        print("-" * 50)
        
        self.model = LightMLP(
            input_size=self.config['input_size'],
            random_state=self.config['random_state']
        )
        
        print(f"\n⚙️  Configuration du modèle:")
        print(f"   • Architecture: {self.config['input_size']} → 16 → 8 → 1")
        print(f"   • Paramètres: {self.model.count_parameters():,}")
        print(f"   • Dropout: {self.config['dropout_rate']}")
        print(f"   • L2 lambda: {self.config['l2_lambda']}")
        
        # ------------------------------------------------------------
        # 3. ENTRAÎNEMENT
        # ------------------------------------------------------------
        print("\n🚀 PHASE 3: ENTRAÎNEMENT")
        print("-" * 50)
        
        training_start = time.time()
        self._train_model(X_train, y_train, X_val, y_val)
        training_time = time.time() - training_start
        
        print(f"\n⏱️  Temps d'entraînement: {training_time:.1f} secondes")
        
        # ------------------------------------------------------------
        # 4. ÉVALUATION
        # ------------------------------------------------------------
        print("\n📈 PHASE 4: ÉVALUATION")
        print("-" * 50)
        
        # Évaluation sur validation
        val_metrics = self._evaluate_model(X_val, y_val, "VALIDATION")
        
        # Évaluation sur test
        test_metrics = self._evaluate_model(X_test, y_test, "TEST")
        
        # Analyse d'importance des features
        feature_importance = self._analyze_feature_importance(
            X_test, y_test, self.data_pipeline.SELECTED_FEATURES
        )
        
        # ------------------------------------------------------------
        # 5. VISUALISATION
        # ------------------------------------------------------------
        print("\n📊 PHASE 5: VISUALISATION")
        print("-" * 50)
        
        self._create_visualizations(
            val_metrics, test_metrics, feature_importance, feature_analysis
        )
        
        # ------------------------------------------------------------
        # 6. SAUVEGARDE
        # ------------------------------------------------------------
        print("\n💾 PHASE 6: SAUVEGARDE")
        print("-" * 50)
        
        self._save_artifacts(test_metrics, feature_importance, training_time)
        
        # ------------------------------------------------------------
        # 7. RAPPORT FINAL
        # ------------------------------------------------------------
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE TERMINÉE AVEC SUCCÈS!")
        print("=" * 70)
        
        print(f"\n📊 RÉSUMÉ DES PERFORMANCES:")
        print(f"   • Test Accuracy:  {test_metrics['accuracy']:.3f}")
        print(f"   • Test F1-score:  {test_metrics['f1_score']:.3f}")
        print(f"   • Test AUC:       {test_metrics['auc']:.3f}")
        print(f"   • Temps total:    {total_time:.1f}s")
        
        print(f"\n🏆 FEATURES LES PLUS IMPORTANTES:")
        for feature, importance in feature_importance[:3]:
            print(f"   • {feature}: {importance:.4f}")
        
        print(f"\n💡 RECOMMANDATIONS:")
        if test_metrics['f1_score'] > 0.75:
            print("   ✅ Modèle performant, prêt pour le déploiement")
        elif test_metrics['f1_score'] > 0.65:
            print("   ⚠️  Modèle acceptable, peut être amélioré")
        else:
            print("   ❌ Modèle sous-performant, revoir les features")
        
        return test_metrics
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray):
        """
        Entraîne le modèle avec early stopping
        """
        print(f"\n📚 Données d'entraînement:")
        print(f"   • Train: {X_train.shape[0]:,} échantillons")
        print(f"   • Validation: {X_val.shape[0]:,} échantillons")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / self.config['batch_size']))
        
        print(f"\n⚙️  Hyperparamètres:")
        print(f"   • Epochs: {self.config['epochs']}")
        print(f"   • Batch size: {self.config['batch_size']}")
        print(f"   • Learning rate: {self.config['learning_rate']}")
        print(f"   • Early stopping patience: {self.config['early_stopping_patience']}")
        
        print("\n🚀 Début de l'entraînement...")
        print("-" * 60)
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Mélanger les données
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            iteration = epoch * n_batches + 1
            
            # Entraînement par batch
            for i in range(0, n_samples, self.config['batch_size']):
                X_batch = X_shuffled[i:i+self.config['batch_size']]
                y_batch = y_shuffled[i:i+self.config['batch_size']]
                
                # Forward + backward + update
                batch_loss = self.model.train_epoch(
                    X_batch, y_batch, 
                    self.config['learning_rate'], 
                    iteration
                )
                
                epoch_loss += batch_loss
                iteration += 1
            
            # Loss moyenne
            train_loss = epoch_loss / n_batches
            
            # Évaluation sur validation
            val_loss, val_metrics = self.model.evaluate(X_val, y_val)
            
            # Sauvegarder dans l'historique
            self.model.history['train_loss'].append(train_loss)
            self.model.history['val_loss'].append(val_loss)
            self.model.history['train_acc'].append(val_metrics['accuracy'])
            self.model.history['val_acc'].append(val_metrics['accuracy'])
            self.model.history['train_f1'].append(val_metrics['f1_score'])
            self.model.history['val_f1'].append(val_metrics['f1_score'])
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.model.get_weights()
            else:
                patience_counter += 1
            
            # Affichage progress
            if (epoch + 1) % 20 == 0 or epoch == 0 or patience_counter >= self.config['early_stopping_patience']:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch + 1:3d}/{self.config['epochs']} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.3f} | "
                      f"Val F1: {val_metrics['f1_score']:.3f} | "
                      f"Patience: {patience_counter}/{self.config['early_stopping_patience']}")
            
            # Arrêt prématuré
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"\n⏹️  Early stopping à l'epoch {epoch + 1}")
                # Restaurer les meilleurs poids
                self.model.set_weights(best_weights)
                break
        
        print("-" * 60)
        print("✅ Entraînement terminé!")
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray, dataset_name: str) -> Dict:
        """
        Évalue le modèle sur un ensemble de données
        """
        print(f"\n📊 Évaluation sur {dataset_name}:")
        print(f"   • Échantillons: {X.shape[0]:,}")
        
        # Prédictions
        y_pred = self.model.predict(X, threshold=self.config['threshold'])
        y_prob = self.model.predict_proba(X)
        
        # Métriques
        metrics = self.model.compute_metrics(y, y_pred, y_prob)
        
        # Affichage
        cm = metrics['confusion_matrix']
        print(f"   • Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   • F1-score:  {metrics['f1_score']:.4f}")
        print(f"   • Precision: {metrics['precision']:.4f}")
        print(f"   • Recall:    {metrics['recall']:.4f}")
        print(f"   • AUC:       {metrics['auc']:.4f}")
        print(f"\n   • Matrice de confusion:")
        print(f"       TN={cm['TN']:4d}  FP={cm['FP']:4d}")
        print(f"       FN={cm['FN']:4d}  TP={cm['TP']:4d}")
        
        # Taux d'erreur par classe
        error_rate_class_0 = cm['FP'] / (cm['TN'] + cm['FP']) if (cm['TN'] + cm['FP']) > 0 else 0
        error_rate_class_1 = cm['FN'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
        
        print(f"   • Taux d'erreur classe 0: {error_rate_class_0:.3f}")
        print(f"   • Taux d'erreur classe 1: {error_rate_class_1:.3f}")
        
        return metrics
    
    def _analyze_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                   feature_names: list) -> list:
        """
        Analyse l'importance des features
        """
        print(f"\n🔍 Analyse d'importance des features...")
        
        # Importance par permutation
        importance_dict = self.model.get_feature_importance(
            X, y, feature_names, n_permutations=5
        )
        
        # Trier par importance
        sorted_importance = sorted(
            importance_dict.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        print(f"\n📊 Classement des features:")
        for i, (feature, importance) in enumerate(sorted_importance, 1):
            importance_str = "🔥" if abs(importance) > 0.05 else "✅" if abs(importance) > 0.02 else "⚠️"
            print(f"   {i:2d}. {importance_str} {feature:25s}: {importance:+.4f}")
        
        return sorted_importance
    
    def _create_visualizations(self, val_metrics: Dict, test_metrics: Dict,
                              feature_importance: list, feature_analysis: Dict):
        """
        Crée les visualisations des résultats
        """
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('ANALYSE DU MODÈLE MLP POUR LA PRÉDICTION DE RÉUSSITE',
                    fontsize=16, fontweight='bold', y=1.02)
        
        # 1. Courbes d'apprentissage
        ax1 = plt.subplot(2, 3, 1)
        epochs = range(1, len(self.model.history['train_loss']) + 1)
        ax1.plot(epochs, self.model.history['train_loss'], 
                label='Train Loss', linewidth=2, color='blue')
        ax1.plot(epochs, self.model.history['val_loss'], 
                label='Val Loss', linewidth=2, color='orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Courbes d\'Apprentissage', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Accuracy pendant l'entraînement
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(epochs, self.model.history['train_acc'], 
                label='Train Accuracy', linewidth=2, color='green')
        ax2.plot(epochs, self.model.history['val_acc'], 
                label='Val Accuracy', linewidth=2, color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy pendant l\'Entraînement', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Importance des features
        ax3 = plt.subplot(2, 3, 3)
        features = [f[0] for f in feature_importance]
        importances = [f[1] for f in feature_importance]
        colors = ['#FF6B6B' if imp > 0 else '#4ECDC4' for imp in importances]
        bars = ax3.barh(range(len(features)), importances, color=colors, edgecolor='black')
        ax3.set_yticks(range(len(features)))
        ax3.set_yticklabels(features, fontsize=9)
        ax3.set_xlabel('Impact sur le F1-score')
        ax3.set_title('Importance des Features (Permutation)', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Matrice de confusion (Test)
        ax4 = plt.subplot(2, 3, 4)
        cm = test_metrics['confusion_matrix']
        cm_matrix = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
        im = ax4.imshow(cm_matrix, cmap='Blues', interpolation='nearest', aspect='auto')
        
        # Annotations
        for i in range(2):
            for j in range(2):
                text_color = 'white' if cm_matrix[i, j] > cm_matrix.max() / 2 else 'black'
                ax4.text(j, i, f"{cm_matrix[i, j]:,}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')
        
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['Prédit 0', 'Prédit 1'])
        ax4.set_yticklabels(['Réel 0', 'Réel 1'])
        ax4.set_title('Matrice de Confusion (Test)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax4)
        
        # 5. Métriques comparées
        ax5 = plt.subplot(2, 3, 5)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
        val_values = [
            val_metrics['accuracy'], val_metrics['precision'],
            val_metrics['recall'], val_metrics['f1_score'], val_metrics['auc']
        ]
        test_values = [
            test_metrics['accuracy'], test_metrics['precision'],
            test_metrics['recall'], test_metrics['f1_score'], test_metrics['auc']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax5.bar(x - width/2, val_values, width, label='Validation', color='skyblue', edgecolor='black')
        ax5.bar(x + width/2, test_values, width, label='Test', color='lightcoral', edgecolor='black')
        
        ax5.set_xlabel('Métriques')
        ax5.set_ylabel('Score')
        ax5.set_title('Comparaison des Métriques', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Résumé statistique
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Texte de résumé
        summary_text = f"""
        🏆 RÉSUMÉ DU MODÈLE
        
        📊 Performances Test:
        • Accuracy:  {test_metrics['accuracy']:.3f}
        • F1-score:  {test_metrics['f1_score']:.3f}
        • Precision: {test_metrics['precision']:.3f}
        • Recall:    {test_metrics['recall']:.3f}
        • AUC:       {test_metrics['auc']:.3f}
        
        🔍 Top 3 Features:
        1. {feature_importance[0][0]}: {feature_importance[0][1]:+.4f}
        2. {feature_importance[1][0]}: {feature_importance[1][1]:+.4f}
        3. {feature_importance[2][0]}: {feature_importance[2][1]:+.4f}
        
        ⚙️ Configuration:
        • Architecture: 7 → 16 → 8 → 1
        • Paramètres: {self.model.count_parameters():,}
        • Dropout: {self.config['dropout_rate']}
        • L2: {self.config['l2_lambda']}
        
        📈 Résultat:
        • {'✅ EXCELLENT' if test_metrics['f1_score'] > 0.75 else '⚠️ ACCEPTABLE' if test_metrics['f1_score'] > 0.65 else '❌ À AMÉLIORER'}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=9, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                         edgecolor='#ddd', pad=10))
        
        plt.tight_layout()
        plt.savefig('model_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Visualisations sauvegardées: model_analysis.png")
    
    def _save_artifacts(self, test_metrics: Dict, feature_importance: list, 
                       training_time: float):
        """
        Sauvegarde tous les artefacts du modèle
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Sauvegarder le modèle
        model_path = f'mlp_reussite_model_{timestamp}.npz'
        self.model.save_model(model_path)
        
        # 2. Sauvegarder le pipeline de données
        pipeline_path = f'data_pipeline_{timestamp}.npz'
        self.data_pipeline.save_pipeline(pipeline_path)
        
        # 3. Créer un rapport détaillé
        report = {
            'timestamp': timestamp,
            'model_info': {
                'architecture': '7_16_8_1',
                'parameters': self.model.count_parameters(),
                'training_time_seconds': round(training_time, 2),
                'final_epochs': len(self.model.history['train_loss'])
            },
            'hyperparameters': self.config,
            'test_performance': test_metrics,
            'feature_importance': {
                feature: float(importance) 
                for feature, importance in feature_importance
            },
            'data_info': {
                'selected_features': self.data_pipeline.SELECTED_FEATURES,
                'excluded_features': self.data_pipeline.EXCLUDED_FEATURES
            },
            'recommendations': self._generate_recommendations(test_metrics, feature_importance)
        }
        
        # Sauvegarder le rapport
        report_path = f'training_report_{timestamp}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Artefacts sauvegardés:")
        print(f"   • Modèle: {model_path}")
        print(f"   • Pipeline: {pipeline_path}")
        print(f"   • Rapport: {report_path}")
        print(f"   • Visualisation: model_analysis.png")
    
    def _generate_recommendations(self, test_metrics: Dict, feature_importance: list) -> list:
        """
        Génère des recommandations basées sur les résultats
        """
        recommendations = []
        
        # Évaluation des performances
        f1_score = test_metrics['f1_score']
        
        if f1_score > 0.75:
            recommendations.append("✅ Le modèle a d'excellentes performances, prêt pour le déploiement.")
        elif f1_score > 0.65:
            recommendations.append("⚠️  Le modèle a des performances acceptables, peut être amélioré avec plus de données.")
        else:
            recommendations.append("❌ Le modèle sous-performe, revoir la sélection des features.")
        
        # Analyse des features
        top_feature, top_importance = feature_importance[0]
        
        if abs(top_importance) > 0.05:
            recommendations.append(f"🔥 La feature '{top_feature}' est très importante, confirme notre analyse.")
        
        # Vérification du sur-apprentissage
        if len(self.model.history['train_loss']) > 0:
            train_loss_final = self.model.history['train_loss'][-1]
            val_loss_final = self.model.history['val_loss'][-1]
            
            if val_loss_final > train_loss_final * 1.2:
                recommendations.append("⚠️  Signes de sur-apprentissage, augmenter la régularisation.")
            else:
                recommendations.append("✅ Bonne généralisation, pas de sur-apprentissage détecté.")
        
        # Recommandations techniques
        recommendations.append("💡 Pour la production: quantifier les poids en INT8 pour réduire la taille.")
        recommendations.append("💡 Surveillance: suivre la dérive des données en production.")
        
        return recommendations
    
    def predict_single(self, student_data: dict) -> dict:
        """
        Prédit pour un seul étudiant
        
        Args:
            student_data: Dictionnaire avec les features
            
        Returns:
            Dictionnaire avec la prédiction et l'analyse
        """
        # Vérifier que le modèle et le pipeline sont chargés
        if self.model is None or self.data_pipeline is None:
            raise ValueError("Le modèle et le pipeline doivent être initialisés d'abord.")
        
        # Convertir en tableau numpy
        features_array = []
        for feature in self.data_pipeline.SELECTED_FEATURES:
            if feature in student_data:
                features_array.append(student_data[feature])
            else:
                # Valeur par défaut (moyenne)
                if feature in self.data_pipeline.feature_stats:
                    features_array.append(self.data_pipeline.feature_stats['means'][
                        self.data_pipeline.SELECTED_FEATURES.index(feature)
                    ])
                else:
                    features_array.append(0.5)
        
        X_new = np.array([features_array], dtype=np.float32)
        
        # Normaliser
        X_scaled = self.data_pipeline.normalize_new_data(X_new)
        
        # Prédiction
        probability = self.model.predict_proba(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]
        
        # Analyse détaillée
        analysis = {
            'probability': float(probability),
            'prediction': int(prediction),
            'class': 'Réussite' if prediction == 1 else 'Échec',
            'confidence': self._get_confidence_level(probability),
            'features_analysis': {}
        }
        
        # Analyse par feature
        for i, feature in enumerate(self.data_pipeline.SELECTED_FEATURES):
            value = student_data.get(feature, 0.5)
            normalized_value = X_scaled[0, i]
            
            # Évaluer si la valeur est favorable
            is_favorable = normalized_value > 0.6  # Supérieur à 60e percentile
            
            analysis['features_analysis'][feature] = {
                'value': float(value),
                'normalized_value': float(normalized_value),
                'is_favorable': bool(is_favorable),
                'contribution': 'positive' if is_favorable else 'negative'
            }
        
        return analysis
    
    def _get_confidence_level(self, probability: float) -> str:
        """Détermine le niveau de confiance de la prédiction"""
        distance_from_05 = abs(probability - 0.5)
        
        if distance_from_05 > 0.3:
            return 'Élevée'
        elif distance_from_05 > 0.15:
            return 'Modérée'
        else:
            return 'Faible'


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    """
    Script principal d'exécution
    """
    
    print("=" * 70)
    print("🏫 MLP POUR LA PRÉDICTION DE RÉUSSITE ACADÉMIQUE")
    print("=" * 70)
    
    try:
        # Initialiser la pipeline
        pipeline = TrainingPipeline()
        
        # Exécuter la pipeline complète
        test_metrics = pipeline.run()
        
        # Exemple de prédiction sur un nouvel étudiant
        print("\n" + "=" * 70)
        print("🎓 EXEMPLE DE PRÉDICTION SUR UN NOUVEL ÉTUDIANT")
        print("=" * 70)
        
        # Profil type d'un bon étudiant
        good_student = {
            'Niveau_etude': 0.8,
            'Heures_etude_ordinal': 3.0,  # Plus de 10h
            'Planning_ordinal': 3.0,      # Oui, toujours
            'Assiduite_ordinal': 3.0,     # Oui, presque toujours
            'Environnement_ordinal': 2.0, # Oui
            'Sommeil_score': 4.0,         # 7-8h, sport régulier
            'Qualite_ordinal': 3.0        # Excellente
        }
        
        # Profil type d'un étudiant à risque
        risky_student = {
            'Niveau_etude': 0.4,
            'Heures_etude_ordinal': 1.0,  # 3-6h
            'Planning_ordinal': 0.0,      # Non
            'Assiduite_ordinal': 1.0,     # Parfois
            'Environnement_ordinal': 0.0, # Non
            'Sommeil_score': 1.0,         # 5-6h, pas de sport
            'Qualite_ordinal': 1.0        # Moyenne
        }
        
        for student_name, student_data in [("Bon étudiant", good_student), 
                                          ("Étudiant à risque", risky_student)]:
            print(f"\n📋 {student_name}:")
            
            prediction = pipeline.predict_single(student_data)
            
            print(f"   → Probabilité de réussite: {prediction['probability']:.1%}")
            print(f"   → Prédiction: {prediction['class']}")
            print(f"   → Confiance: {prediction['confidence']}")
            
            # Afficher les 3 features les plus influentes
            print(f"   → Analyse des features:")
            
            features_analysis = prediction['features_analysis']
            sorted_features = sorted(
                features_analysis.items(),
                key=lambda x: abs(x[1]['normalized_value'] - 0.5),
                reverse=True
            )
            
            for feature, analysis in sorted_features[:3]:
                status = "✅ FAVORABLE" if analysis['is_favorable'] else "❌ DÉFAVORABLE"
                print(f"     • {feature}: {analysis['value']:.2f} ({status})")
        
        print("\n" + "=" * 70)
        print("✅ PROGRAMME TERMINÉ AVEC SUCCÈS!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()