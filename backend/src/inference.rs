use tract_onnx::prelude::*;
use anyhow::{Result, Context};
use std::sync::Arc;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use dashmap::DashMap;
use std::time::{Instant, Duration};
use once_cell::sync::Lazy;

use crate::models::{StudentFeatures, PredictionResult};

// Cache LRU pour les prédictions fréquentes
static PREDICTION_CACHE: Lazy<DashMap<[u8; 28], (PredictionResult, Instant)>> = Lazy::new(|| {
    DashMap::with_capacity(1000)
});

// Statistiques d'exécution
static TOTAL_PREDICTIONS: AtomicU64 = AtomicU64::new(0);
static TOTAL_BATCH_SIZE: AtomicU64 = AtomicU64::new(0);
static AVG_LATENCY_MS: AtomicU64 = AtomicU64::new(0);

pub struct ModelInference {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    threshold: f32,
}

impl ModelInference {
    pub fn load<P: AsRef<Path>>(model_path: P, threshold: f32) -> Result<Self> {
        log::info!("Chargement du modèle ONNX: {:?}", model_path.as_ref());
        
        let model_path_ref = model_path.as_ref();
        
        // Vérification du fichier
        if !model_path_ref.exists() {
            return Err(anyhow::anyhow!("Fichier modèle non trouvé: {:?}", model_path_ref));
        }
        
        // Charger avec shape dynamique pour batch (-1 pour dimension variable)
        let model = tract_onnx::onnx()
            .model_for_path(model_path_ref)
            .with_context(|| format!("Échec du chargement du modèle depuis {:?}", model_path_ref))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec!(TDim::s(), TDim::from(7)),  // S pour dynamique
                ),
            )?
            .into_optimized()?
            .into_runnable()?;
        
        log::info!("✅ Modèle chargé avec succès (seuil: {}, batch dynamique activé)", threshold);
        
        Ok(ModelInference { model, threshold })
    }
    
    // Prédiction simple optimisée avec cache
    pub fn predict(&self, features: &StudentFeatures) -> Result<PredictionResult> {
        let start_time = Instant::now();
        
        // Vérifier le cache
        let features_hash = Self::hash_features(features);
        if let Some(cached) = PREDICTION_CACHE.get(&features_hash) {
            let (result, timestamp) = cached.value();
            if timestamp.elapsed() < Duration::from_secs(300) {
                log::debug!("Cache hit pour prédiction");
                return Ok(result.clone());
            }
        }
        
        // Validation
        if let Err(e) = features.validate() {
            return Err(anyhow::anyhow!("Validation des features échouée: {}", e));
        }
        
        let features_array = features.to_array();
        
        // Création du tenseur
        let input_tensor = tract_onnx::prelude::Tensor::from_shape(&[1, 7], &features_array)
            .context("Échec de la création du tenseur d'entrée")?;
        
        // Exécution du modèle
        let outputs: TVec<TValue> = self.model
            .run(tvec!(input_tensor.into()))
            .context("Échec de l'exécution du modèle")?;
        
        // Extraire la probabilité
        let probability: f32 = outputs[0]
            .clone()
            .into_tensor()
            .to_array_view::<f32>()
            .context("Échec de la conversion des sorties")?
            .into_iter()
            .next()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Aucune sortie du modèle"))?;
        
        // Clamp pour sécurité
        let probability = probability.clamp(0.0, 1.0);
        
        // Création du résultat
        let result = PredictionResult::new(probability, features, self.threshold);
        
        // Mettre en cache
        PREDICTION_CACHE.insert(features_hash, (result.clone(), Instant::now()));
        
        // Mettre à jour les statistiques
        let latency_ms = start_time.elapsed().as_millis() as u64;
        TOTAL_PREDICTIONS.fetch_add(1, Ordering::Relaxed);
        AVG_LATENCY_MS.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
            Some((old + latency_ms) / 2)
        });
        
        log::info!("Prédiction: prob={:.3}, succès={}, latence={}ms", 
            probability, result.success, latency_ms);
        
        Ok(result)
    }
    
    // Prédiction batch OPTIMISÉE
    pub fn batch_predict(&self, features_list: &[StudentFeatures]) -> Result<Vec<PredictionResult>> {
        let start_time = Instant::now();
        
        if features_list.is_empty() {
            return Ok(Vec::new());
        }
        
        // Validation batch
        for (i, features) in features_list.iter().enumerate() {
            if let Err(e) = features.validate() {
                return Err(anyhow::anyhow!("Étudiant {}: {}", i + 1, e));
            }
        }
        
        // Préparer le batch tensor
        let batch_size = features_list.len();
        let mut batch_data = Vec::with_capacity(batch_size * 7);
        
        for features in features_list {
            let arr = features.to_array();
            batch_data.extend_from_slice(&arr);
        }
        
        // Créer le tenseur batch
        let input_tensor = tract_onnx::prelude::Tensor::from_shape(
            &[batch_size as i64, 7], 
            &batch_data
        )
        .context("Échec de la création du tenseur batch")?;
        
        // Exécution batch
        let outputs: TVec<TValue> = self.model
            .run(tvec!(input_tensor.into()))
            .context("Échec de l'exécution du modèle batch")?;
        
        let output_tensor = outputs[0]
            .clone()
            .into_tensor();
        
        let probabilities = output_tensor
            .to_array_view::<f32>()
            .context("Échec de la conversion des sorties batch")?
            .into_dimensionality::<ndarray::Ix1>()
            .context("Échec du reshape des sorties batch")?;
        
        // Créer les résultats
        let results: Vec<PredictionResult> = features_list.iter()
            .zip(probabilities.iter())
            .map(|(features, &prob)| {
                let prob = prob.clamp(0.0, 1.0);
                PredictionResult::new(prob, features, self.threshold)
            })
            .collect();
        
        // Mettre à jour les statistiques
        let latency_ms = start_time.elapsed().as_millis() as u64;
        TOTAL_PREDICTIONS.fetch_add(batch_size as u64, Ordering::Relaxed);
        TOTAL_BATCH_SIZE.fetch_add(batch_size as u64, Ordering::Relaxed);
        
        log::info!("Batch: {} étudiants, latence={}ms, avg={}ms/étudiant", 
            batch_size, latency_ms, latency_ms as f32 / batch_size as f32);
        
        Ok(results)
    }
    
    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            input_shape: vec![-1, 7],  // -1 pour batch dynamique
            threshold: self.threshold,
            version: "1.0.0".to_string(),
            features: vec![
                "Niveau d'étude".to_string(),
                "Heures d'étude".to_string(),
                "Planning".to_string(),
                "Assiduité".to_string(),
                "Environnement".to_string(),
                "Sommeil".to_string(),
                "Qualité d'étude".to_string(),
            ],
            supports_batch: true,
            cache_size: PREDICTION_CACHE.len() as u64,
            total_predictions: TOTAL_PREDICTIONS.load(Ordering::Relaxed),
            avg_latency_ms: AVG_LATENCY_MS.load(Ordering::Relaxed),
        }
    }
    
    // Fonctions utilitaires
    fn hash_features(features: &StudentFeatures) -> [u8; 28] {
        use std::mem;
        let arr = features.to_array();
        unsafe { mem::transmute(arr) }
    }
    
    // Nettoyage du cache
    pub fn clear_cache(&self) -> usize {
        let before = PREDICTION_CACHE.len();
        PREDICTION_CACHE.retain(|_, (_, timestamp)| {
            timestamp.elapsed() < Duration::from_secs(300)
        });
        let after = PREDICTION_CACHE.len();
        before - after
    }
}

#[derive(Debug, serde::Serialize)]
pub struct ModelInfo {
    pub input_shape: Vec<i32>,
    pub threshold: f32,
    pub version: String,
    pub features: Vec<String>,
    pub supports_batch: bool,
    pub cache_size: u64,
    pub total_predictions: u64,
    pub avg_latency_ms: u64,
}

// Modèle global
static MODEL: Lazy<Result<Arc<ModelInference>, String>> = Lazy::new(|| {
    match ModelInference::load("student_model.onnx", 0.61) {
        Ok(model) => {
            log::info!("✅ Modèle ONNX chargé avec succès");
            Ok(Arc::new(model))
        }
        Err(e) => Err(format!("Échec du chargement du modèle: {}", e)),
    }
});

pub fn get_model() -> Result<Arc<ModelInference>, String> {
    match &*MODEL {
        Ok(model) => Ok(model.clone()),
        Err(e) => Err(e.clone()),
    }
}

// Statistiques globales
pub fn get_stats() -> InferenceStats {
    InferenceStats {
        cache_size: PREDICTION_CACHE.len() as u64,
        total_predictions: TOTAL_PREDICTIONS.load(Ordering::Relaxed),
        total_batch_size: TOTAL_BATCH_SIZE.load(Ordering::Relaxed),
        avg_latency_ms: AVG_LATENCY_MS.load(Ordering::Relaxed),
    }
}

#[derive(Debug, serde::Serialize)]
pub struct InferenceStats {
    pub cache_size: u64,
    pub total_predictions: u64,
    pub total_batch_size: u64,
    pub avg_latency_ms: u64,
}