use tract_onnx::prelude::*; // Arc supprimé car inutilisé
use serde::Serialize;

#[derive(Clone)]
pub struct ModelInference {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    pub threshold: f32,
}

impl ModelInference {
    pub fn load<P: AsRef<std::path::Path>>(model_path: P, threshold: f32) -> TractResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 7)),
            )?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self { model, threshold })
    }

    pub fn predict(&self, features: &[f32; 7]) -> TractResult<f32> {
        let input_tensor = tract_onnx::prelude::Tensor::from_shape(&[1, 7], features)?;
        let outputs = self.model.run(tvec!(input_tensor.into()))?;
        
        // Correction de l'erreur d'indexation
        let logits: f32 = *outputs[0].to_array_view::<f32>()?.iter().next()
            .ok_or_else(|| anyhow::anyhow!("Aucune sortie du modèle"))?;
        
        Ok(1.0 / (1.0 + (-logits).exp())) // sigmoid
    }

    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            input_shape: vec![1, 7],
            threshold: self.threshold,
            version: "2.0.0".to_string(),
            features: vec![
                "Niveau d'étude".to_string(),
                "Heures d'étude".to_string(),
                "Planning".to_string(),
                "Assiduité".to_string(),
                "Environnement".to_string(),
                "Sommeil".to_string(),
                "Qualité d'étude".to_string(),
            ],
        }
    }
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub input_shape: Vec<i32>,
    pub threshold: f32,
    pub version: String,
    pub features: Vec<String>,
}
