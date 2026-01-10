pub mod models;
pub mod inference;

use actix_web::{web, App, HttpServer, HttpResponse, Responder, HttpRequest};
use actix_files::{Files, NamedFile};
use actix_cors::Cors;
use actix_web::middleware::{Logger, DefaultHeaders};
use std::sync::Arc;
use std::time::Instant;
use log::{info, error, warn};
use inference::get_model;
use models::{StudentFeatures, ApiResponse, PredictionResult};

// Middleware de sécurité
const API_KEY_HEADER: &str = "X-API-Key";
const VALID_API_KEYS: &[&str] = &["prod_key_123", "test_key_456"];

// Rate limiting simplifié avec dashmap
use dashmap::DashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

static RATE_LIMIT_CACHE: once_cell::sync::Lazy<DashMap<String, (AtomicU32, Instant)>> = 
    once_cell::sync::Lazy::new(|| DashMap::new());

// Handlers optimisés
async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(ApiResponse::success("✅ API de prédiction étudiante"))
}

async fn model_info() -> impl Responder {
    match get_model() {
        Ok(model) => {
            let info = model.get_model_info();
            HttpResponse::Ok().json(ApiResponse::success(info))
        }
        Err(e) => {
            error!("Erreur récupération info modèle: {}", e);
            HttpResponse::InternalServerError().json(ApiResponse::<String>::error(&e))
        }
    }
}

async fn predict_success(
    model: web::Data<Arc<inference::ModelInference>>,
    req: web::Json<StudentFeatures>,
    request: HttpRequest,
) -> impl Responder {
    let start_time = Instant::now();
    
    // Rate limiting simplifié
    if let Some(client_ip) = request.peer_addr().map(|addr| addr.ip().to_string()) {
        let now = Instant::now();
        let entry = RATE_LIMIT_CACHE
            .entry(client_ip.clone())
            .or_insert_with(|| (AtomicU32::new(0), now));
        
        // Nettoyer si plus d'une minute
        if now.duration_since(entry.1) > Duration::from_secs(60) {
            entry.0.store(0, Ordering::Relaxed);
            entry.1 = now;
        }
        
        let count = entry.0.fetch_add(1, Ordering::Relaxed);
        if count > 100 {
            warn!("Rate limit dépassé pour IP: {}", client_ip);
            return HttpResponse::TooManyRequests()
                .json(ApiResponse::<PredictionResult>::error("Rate limit dépassé"));
        }
    }
    
    info!("Nouvelle requête de prédiction reçue");
    
    if let Err(e) = req.validate() {
        error!("Validation échouée: {}", e);
        let mut response = ApiResponse::<PredictionResult>::error(&e);
        response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
        return HttpResponse::BadRequest().json(response);
    }
    
    let model_clone = model.clone();
    let features = req.into_inner();
    
    match web::block(move || model_clone.predict(&features)).await {
        Ok(result) => match result {
            Ok(result) => {
                info!("Prédiction réussie: probabilité={:.3}", result.probability);
                let mut response = ApiResponse::success(result);
                response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
                HttpResponse::Ok().json(response)
            }
            Err(e) => {
                error!("Erreur de prédiction: {}", e);
                let mut response = ApiResponse::<PredictionResult>::error(&format!("Erreur interne: {}", e));
                response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
                HttpResponse::InternalServerError().json(response)
            }
        },
        Err(e) => {
            error!("Erreur d'exécution bloquante: {}", e);
            let mut response = ApiResponse::<PredictionResult>::error("Erreur d'exécution");
            response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
            HttpResponse::InternalServerError().json(response)
        }
    }
}

async fn batch_predict(
    model: web::Data<Arc<inference::ModelInference>>,
    req: web::Json<Vec<StudentFeatures>>,
    request: HttpRequest,
) -> impl Responder {
    let start_time = Instant::now();
    
    // Rate limiting simplifié
    if let Some(client_ip) = request.peer_addr().map(|addr| addr.ip().to_string()) {
        let now = Instant::now();
        let entry = RATE_LIMIT_CACHE
            .entry(client_ip.clone())
            .or_insert_with(|| (AtomicU32::new(0), now));
        
        if now.duration_since(entry.1) > Duration::from_secs(60) {
            entry.0.store(0, Ordering::Relaxed);
            entry.1 = now;
        }
        
        let count = entry.0.fetch_add(1, Ordering::Relaxed);
        if count > 20 {
            warn!("Rate limit batch dépassé pour IP: {}", client_ip);
            let mut response = ApiResponse::<Vec<PredictionResult>>::error("Rate limit dépassé");
            response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
            return HttpResponse::TooManyRequests().json(response);
        }
    }
    
    info!("Nouvelle requête de batch prediction: {} étudiants", req.len());
    
    if req.is_empty() {
        let mut response = ApiResponse::<Vec<PredictionResult>>::error("Liste d'étudiants vide");
        response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
        return HttpResponse::BadRequest().json(response);
    }
    
    for (i, features) in req.iter().enumerate() {
        if let Err(e) = features.validate() {
            let mut response = ApiResponse::<Vec<PredictionResult>>::error(
                &format!("Étudiant {}: {}", i + 1, e)
            );
            response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
            return HttpResponse::BadRequest().json(response);
        }
    }
    
    let model_clone = model.clone();
    let features_list = req.into_inner();
    
    match web::block(move || model_clone.batch_predict(&features_list)).await {
        Ok(result) => match result {
            Ok(results) => {
                info!("Batch prédiction réussie: {} résultats", results.len());
                let mut response = ApiResponse::success(results);
                response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
                HttpResponse::Ok().json(response)
            }
            Err(e) => {
                error!("Erreur batch prédiction: {}", e);
                let mut response = ApiResponse::<Vec<PredictionResult>>::error(
                    &format!("Erreur interne: {}", e)
                );
                response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
                HttpResponse::InternalServerError().json(response)
            }
        },
        Err(e) => {
            error!("Erreur d'exécution bloquante batch: {}", e);
            let mut response = ApiResponse::<Vec<PredictionResult>>::error("Erreur d'exécution");
            response.execution_time_ms = Some(start_time.elapsed().as_millis() as u64);
            HttpResponse::InternalServerError().json(response)
        }
    }
}

async fn stats() -> impl Responder {
    let stats = inference::get_stats();
    HttpResponse::Ok().json(ApiResponse::success(stats))
}

async fn clear_cache(model: web::Data<Arc<inference::ModelInference>>) -> impl Responder {
    let cleared = model.clear_cache();
    HttpResponse::Ok().json(ApiResponse::success(format!("Cache nettoyé: {} entrées", cleared)))
}

// Interface web
async fn index(req: HttpRequest) -> impl Responder {
    match NamedFile::open_async("./static/index.html").await {
        Ok(file) => file.into_response(&req),
        Err(_) => HttpResponse::InternalServerError().body("Erreur chargement interface")
    }
}

// Middleware de validation API key simplifié
async fn validate_api_key(req: HttpRequest) -> Result<HttpRequest, actix_web::Error> {
    if let Some(api_key) = req.headers().get(API_KEY_HEADER) {
        if VALID_API_KEYS.iter().any(|&key| key == api_key.to_str().unwrap_or("")) {
            Ok(req)
        } else {
            Err(actix_web::error::ErrorUnauthorized("API key invalide"))
        }
    } else {
        Err(actix_web::error::ErrorUnauthorized("API key manquante"))
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Configuration avancée du logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .format_module_path(false)
        .init();
    
    info!("🚀 Démarrage de l'API de Prédiction de Réussite Étudiante");
    
    // Charger le modèle
    let model = match get_model() {
        Ok(model) => {
            info!("✅ Modèle ONNX chargé avec succès");
            model
        }
        Err(e) => {
            error!("❌ {}", e);
            panic!("Impossible de charger le modèle: {}", e);
        }
    };
    
    let model_data = web::Data::new(model);
    
    // Configuration serveur
    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let workers = std::env::var("WORKERS")
        .map(|w| w.parse().unwrap_or(num_cpus::get()))
        .unwrap_or_else(|_| num_cpus::get());
    
    let bind_address = format!("{}:{}", host, port);
    
    info!("🌐 Serveur démarré sur: http://{}", bind_address);
    info!("👷 Workers: {}", workers);
    info!("📊 Interface disponible sur: http://{}/", bind_address);
    info!("🔧 Endpoints API:");
    info!("   GET  /api/health         - Vérification santé");
    info!("   GET  /api/model-info     - Information modèle");
    info!("   GET  /api/stats          - Statistiques");
    info!("   POST /api/predict        - Prédiction simple");
    info!("   POST /api/batch-predict  - Prédiction multiple");
    info!("   POST /api/clear-cache    - Nettoyage cache");
    
    HttpServer::new(move || {
        // Configuration CORS sécurisée
        let cors = Cors::default()
            .allowed_origin("http://localhost:8080")
            .allowed_origin("http://127.0.0.1:8080")
            .allowed_methods(vec!["GET", "POST"])
            .allowed_headers(vec![
                actix_web::http::header::CONTENT_TYPE,
                actix_web::http::header::AUTHORIZATION,
                actix_web::http::header::HeaderName::from_static(API_KEY_HEADER),
            ])
            .max_age(3600);
        
        App::new()
            .wrap(Logger::default())
            .wrap(DefaultHeaders::new().add(("X-Content-Type-Options", "nosniff")))
            .wrap(cors)
            .app_data(model_data.clone())
            .app_data(web::JsonConfig::default().limit(10 * 1024 * 1024))
            // Routes publiques
            .route("/api/health", web::get().to(health_check))
            .route("/api/model-info", web::get().to(model_info))
            .route("/api/stats", web::get().to(stats))
            // Routes protégées avec middleware
            .service(
                web::resource("/api/predict")
                    .wrap_fn(|req, srv| {
                        let api_key_check = validate_api_key(req);
                        async move {
                            match api_key_check.await {
                                Ok(req) => srv.call(req).await,
                                Err(e) => Err(e),
                            }
                        }
                    })
                    .route(web::post().to(predict_success))
            )
            .service(
                web::resource("/api/batch-predict")
                    .wrap_fn(|req, srv| {
                        let api_key_check = validate_api_key(req);
                        async move {
                            match api_key_check.await {
                                Ok(req) => srv.call(req).await,
                                Err(e) => Err(e),
                            }
                        }
                    })
                    .route(web::post().to(batch_predict))
            )
            .service(
                web::resource("/api/clear-cache")
                    .wrap_fn(|req, srv| {
                        let api_key_check = validate_api_key(req);
                        async move {
                            match api_key_check.await {
                                Ok(req) => srv.call(req).await,
                                Err(e) => Err(e),
                            }
                        }
                    })
                    .route(web::post().to(clear_cache))
            )
            // Interface web
            .route("/", web::get().to(index))
            // Fichiers statiques
            .service(Files::new("/static", "./static").prefer_utf8(true))
            .service(Files::new("/favicon.ico", "./static"))
            // Fallback 404
            .default_service(web::route().to(|| async {
                HttpResponse::NotFound().json(ApiResponse::<String>::error("Endpoint non trouvé"))
            }))
    })
    .workers(workers)
    .bind(&bind_address)?
    .run()
    .await
}