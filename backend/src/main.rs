use actix_web::{web, App, HttpServer, HttpResponse, Responder};
use actix_files::Files;
use std::sync::Arc;
use log::{info, error};

mod inference;
mod models;

use inference::ModelInference;
use models::{StudentFeatures, ApiResponse, PredictionResult};

async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(ApiResponse::<String>::success(
        "✅ API de prédiction étudiante en ligne".to_string() // conversion &str -> String
    ))
}

async fn model_info(model: web::Data<Arc<ModelInference>>) -> impl Responder {
    HttpResponse::Ok().json(ApiResponse::success(model.get_model_info()))
}

async fn predict(
    model: web::Data<Arc<ModelInference>>,
    req: web::Json<StudentFeatures>,
) -> impl Responder {
    match model.predict(&req.to_array()) {
        Ok(prob) => {
            let result = PredictionResult::new(prob, &req);
            HttpResponse::Ok().json(ApiResponse::success(result))
        }
        Err(e) => {
            error!("Erreur de prédiction: {}", e);
            HttpResponse::InternalServerError().json(ApiResponse::<String>::error(
                &format!("Erreur interne: {}", e)
            ))
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    info!(" Démarrage de l'API de Prédiction Étudiante");

    let model = Arc::new(ModelInference::load("student_model.onnx", 0.61)
        .expect("Impossible de charger le modèle ONNX"));
    let model_data = web::Data::new(model);

    let host = "127.0.0.1";
    let port = 8080;

    info!(" Serveur démarré sur http://{}:{}/", host, port);

    HttpServer::new(move || {
        App::new()
            .app_data(model_data.clone())
            .route("/api/health", web::get().to(health_check))
            .route("/api/model-info", web::get().to(model_info))
            .route("/api/predict", web::post().to(predict))
            .service(Files::new("/", "./static").index_file("index.html"))
    })
    .bind((host, port))?
    .run()
    .await
}
