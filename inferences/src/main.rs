use actix_web::{App, HttpServer};
use std::sync::Arc;

mod routes;
mod model_inference;
mod types;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model = Arc::new(
        model_inference::InferenceModel::new("models/export_onnx.onnx").unwrap()
    );

    HttpServer::new(move || {
        App::new()
            .app_data(actix_web::web::Data::new(model.clone()))
            .service(routes::predict)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
