use serde::{Deserialize, Serialize};
use chrono::Utc;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StudentFeatures {
    pub niveau_etude: f32,
    pub heures_etude_ordinal: f32,
    pub planning_ordinal: f32,
    pub assiduite_ordinal: f32,
    pub environnement_ordinal: f32,
    pub sommeil_score: f32,
    pub qualite_ordinal: f32,
}

impl StudentFeatures {
    pub fn to_array(&self) -> [f32; 7] {
        [
            self.niveau_etude,
            self.heures_etude_ordinal,
            self.planning_ordinal,
            self.assiduite_ordinal,
            self.environnement_ordinal,
            self.sommeil_score,
            self.qualite_ordinal,
        ]
    }
}

#[derive(Debug, Serialize)]
pub struct FeatureAnalysis {
    pub name: String,
    pub value: f32,
    pub impact: String,
    pub color: String,
}

#[derive(Debug, Serialize)]
pub struct PredictionResult {
    pub probability: f32,
    pub success: bool,
    pub confidence: String,
    pub confidence_color: String,
    pub recommendation: String,
    pub features_analysis: Vec<FeatureAnalysis>,
    pub timestamp: String,
    pub model_version: String,
    pub message: String,
}

impl PredictionResult {
    pub fn new(probability: f32, features: &StudentFeatures) -> Self {
        let success = probability >= 0.61;
        let (confidence, confidence_color) = Self::calculate_confidence(probability);
        let (recommendation, message) = Self::generate_recommendation(probability);

        PredictionResult {
            probability,
            success,
            confidence,
            confidence_color,
            recommendation,
            features_analysis: Self::analyze_features(features),
            timestamp: Utc::now().to_rfc3339(),
            model_version: "2.0.0".to_string(),
            message,
        }
    }

    fn calculate_confidence(prob: f32) -> (String, String) {
        if prob > 0.8 { ("TRÈS ÉLEVÉE".to_string(), "🟢".to_string()) }
        else if prob > 0.6 { ("ÉLEVÉE".to_string(), "🟡".to_string()) }
        else if prob > 0.4 { ("MODÉRÉE".to_string(), "🟠".to_string()) }
        else { ("FAIBLE".to_string(), "🔴".to_string()) }
    }

    fn generate_recommendation(prob: f32) -> (String, String) {
        if prob > 0.8 {
            ("🏆 EXCELLENT".to_string(), "Leadership académique recommandé".to_string())
        } else if prob > 0.65 {
            ("👍 TRÈS BON".to_string(), "Continuité des efforts".to_string())
        } else if prob > 0.5 {
            ("🤔 MODÉRÉ".to_string(), "Focus sur l'amélioration".to_string())
        } else if prob > 0.35 {
            ("⚠️ DIFFICILE".to_string(), "Accompagnement personnalisé".to_string())
        } else {
            ("🚨 CRITIQUE".to_string(), "Plan d'action intensif requis".to_string())
        }
    }

    fn analyze_features(features: &StudentFeatures) -> Vec<FeatureAnalysis> {
        let data = [
            ("Niveau d'étude", features.niveau_etude),
            ("Heures d'étude", features.heures_etude_ordinal),
            ("Planning", features.planning_ordinal),
            ("Assiduité", features.assiduite_ordinal),
            ("Environnement", features.environnement_ordinal),
            ("Sommeil", features.sommeil_score),
            ("Qualité", features.qualite_ordinal),
        ];

        data.iter().map(|(name, value)| {
            let impact = if *value > 0.8 {"EXCELLENT"}
            else if *value > 0.6 {"BON"}
            else if *value > 0.4 {"MOYEN"}
            else if *value > 0.2 {"FAIBLE"}
            else {"TRÈS FAIBLE"};

            let color = if *value > 0.6 {"🟢"} else if *value > 0.4 {"🟡"} else {"🔴"};

            FeatureAnalysis { name: name.to_string(), value: *value, impact: impact.to_string(), color: color.to_string() }
        }).collect()
    }
}

#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: String,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self { success: true, data: Some(data), error: None, timestamp: Utc::now().to_rfc3339() }
    }

    pub fn error(message: &str) -> Self {
        Self { success: false, data: None, error: Some(message.to_string()), timestamp: Utc::now().to_rfc3339() }
    }
}
