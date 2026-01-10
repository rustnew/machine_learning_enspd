use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
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
    pub fn validate(&self) -> Result<(), String> {
        let features = [
            ("Niveau d'étude", self.niveau_etude),
            ("Heures d'étude", self.heures_etude_ordinal),
            ("Planning", self.planning_ordinal),
            ("Assiduité", self.assiduite_ordinal),
            ("Environnement", self.environnement_ordinal),
            ("Sommeil", self.sommeil_score),
            ("Qualité d'étude", self.qualite_ordinal),
        ];

        for (name, value) in features.iter() {
            if !(0.0..=1.0).contains(value) {
                return Err(format!("{} doit être entre 0 et 1 (valeur: {})", name, value));
            }
        }

        Ok(())
    }

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

    pub fn analyze_features(&self) -> Vec<FeatureAnalysis> {
        let features = [
            ("Niveau d'étude", self.niveau_etude),
            ("Heures d'étude", self.heures_etude_ordinal),
            ("Planning", self.planning_ordinal),
            ("Assiduité", self.assiduite_ordinal),
            ("Environnement", self.environnement_ordinal),
            ("Sommeil", self.sommeil_score),
            ("Qualité d'étude", self.qualite_ordinal),
        ];

        features
            .iter()
            .map(|(name, value)| FeatureAnalysis {
                name: name.to_string(),
                value: *value,
                impact: Self::get_impact_text(*value),
                color: Self::get_impact_color(*value),
            })
            .collect()
    }

    fn get_impact_text(value: f32) -> String {
        match value {
            v if v > 0.8 => "EXCELLENT".to_string(),
            v if v > 0.6 => "BON".to_string(),
            v if v > 0.4 => "MOYEN".to_string(),
            v if v > 0.2 => "FAIBLE".to_string(),
            _ => "TRÈS FAIBLE".to_string(),
        }
    }

    fn get_impact_color(value: f32) -> String {
        if value > 0.6 {
            "🟢".to_string()
        } else if value > 0.4 {
            "🟡".to_string()
        } else {
            "🔴".to_string()
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct FeatureAnalysis {
    pub name: String,
    pub value: f32,
    pub impact: String,
    pub color: String,
}

#[derive(Debug, Serialize, Clone)]
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
    pub fn new(probability: f32, features: &StudentFeatures, threshold: f32) -> Self {
        let success = probability >= threshold;
        let (confidence, confidence_color) = Self::calculate_confidence(probability);
        let (recommendation, message) = Self::generate_recommendation(probability);
        
        PredictionResult {
            probability,
            success,
            confidence,
            confidence_color,
            recommendation,
            features_analysis: features.analyze_features(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            model_version: "1.0.0".to_string(),
            message,
        }
    }

    fn calculate_confidence(probability: f32) -> (String, String) {
        let distance_from_mid = (probability - 0.5).abs();
        let confidence_score = 1.0 - distance_from_mid * 2.0;
        
        match confidence_score {
            c if c > 0.8 => ("TRÈS ÉLEVÉE".to_string(), "🟢".to_string()),
            c if c > 0.6 => ("ÉLEVÉE".to_string(), "🟡".to_string()),
            c if c > 0.4 => ("MODÉRÉE".to_string(), "🟠".to_string()),
            _ => ("FAIBLE".to_string(), "🔴".to_string()),
        }
    }

    fn generate_recommendation(probability: f32) -> (String, String) {
        match probability {
            p if p > 0.8 => (
                "🏆 EXCELLENT - Potentiel exceptionnel".to_string(),
                "Leadership académique recommandé".to_string(),
            ),
            p if p > 0.65 => (
                "👍 TRÈS BON - Bonnes perspectives".to_string(),
                "Continuité des efforts actuels".to_string(),
            ),
            p if p > 0.5 => (
                "🤔 MODÉRÉ - Atteignable avec effort".to_string(),
                "Focus sur les points d'amélioration".to_string(),
            ),
            p if p > 0.35 => (
                "⚠️ DIFFICILE - Soutien nécessaire".to_string(),
                "Accompagnement personnalisé recommandé".to_string(),
            ),
            _ => (
                "🚨 CRITIQUE - Intervention urgente".to_string(),
                "Plan d'action intensif requis".to_string(),
            ),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: String,
    pub execution_time_ms: Option<u64>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        ApiResponse {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
            execution_time_ms: None,
        }
    }

    pub fn error(message: &str) -> Self {
        ApiResponse {
            success: false,
            data: None,
            error: Some(message.to_string()),
            timestamp: chrono::Utc::now().to_rfc3339(),
            execution_time_ms: None,
        }
    }
}