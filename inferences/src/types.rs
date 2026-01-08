use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct ModelInput {
    pub features: Vec<f32>, // 7 features
}

#[derive(Serialize)]
pub struct ModelOutput {
    pub prediction: f32,
}
