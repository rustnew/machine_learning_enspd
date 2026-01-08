use actix_web::{post, web, HttpResponse, Responder};
use crate::model_inference::InferenceModel;
use crate::types::{ModelInput, ModelOutput};
use std::sync::Arc;

#[post("/predict")]
pub async fn predict(
    model: web::Data<Arc<InferenceModel>>,
    input: web::Json<ModelInput>
) -> impl Responder {
    match model.predict(input.into_inner()) {
        Ok(output) => HttpResponse::Ok().json(output),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    }
}
