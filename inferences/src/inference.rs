use onnxruntime::{environment::Environment, session::Session, tensor::OrtOwnedTensor, GraphOptimizationLevel};
use crate::types::{ModelInput, ModelOutput};

pub struct InferenceModel {
    session: Session,
}

impl InferenceModel {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let env = Environment::builder()
            .with_name("expot_onnx")
            .build()?;

        let session = env
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_model_from_file(model_path)?;

        Ok(Self { session })
    }

    pub fn predict(&self, input: ModelInput) -> anyhow::Result<ModelOutput> {
        let input_array = ndarray::Array::from_shape_vec((1, input.features.len()), input.features)?;
        let outputs: Vec<OrtOwnedTensor<f32, _>> = self.session.run(vec![input_array.into()])?;

        Ok(ModelOutput {
            prediction: outputs[0][[0,0]],
        })
    }
}
