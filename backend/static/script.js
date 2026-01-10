// Configuration
const API_BASE_URL = window.location.origin;
const MODEL_THRESHOLD = 0.61;

// Éléments DOM
const form = document.getElementById('prediction-form');
const resultsContainer = document.getElementById('results-container');
const loader = document.getElementById('loader');
const notification = document.getElementById('notification');

// Initialisation des sliders
const sliders = [
    'niveau_etude', 'heures_etude_ordinal', 'planning_ordinal',
    'assiduite_ordinal', 'environnement_ordinal', 'sommeil_score', 'qualite_ordinal'
];

// Initialisation
document.addEventListener('DOMContentLoaded', () => {
    initializeSliders();
    setupEventListeners();
    showNotification('Prêt à analyser ! Déplacez les curseurs pour simuler un profil étudiant.', 'info');
});

function initializeSliders() {
    sliders.forEach(sliderId => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(`${sliderId}_value`);
        
        // Initial value
        valueDisplay.textContent = (slider.value / 100).toFixed(2);
        
        // Update on change
        slider.addEventListener('input', () => {
            const value = slider.value / 100;
            valueDisplay.textContent = value.toFixed(2);
        });
    });
}

function setupEventListeners() {
    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await predict();
    });
    
    // Preset buttons
    document.getElementById('btn-preset-good').addEventListener('click', () => setPreset('good'));
    document.getElementById('btn-preset-average').addEventListener('click', () => setPreset('average'));
    document.getElementById('btn-preset-poor').addEventListener('click', () => setPreset('poor'));
    
    // Results actions
    document.getElementById('btn-new-analysis').addEventListener('click', resetForm);
    document.getElementById('btn-share').addEventListener('click', shareResults);
    document.getElementById('btn-export').addEventListener('click', exportResults);
}

function getFeatures() {
    return {
        niveau_etude: parseFloat(document.getElementById('niveau_etude').value) / 100,
        heures_etude_ordinal: parseFloat(document.getElementById('heures_etude_ordinal').value) / 100,
        planning_ordinal: parseFloat(document.getElementById('planning_ordinal').value) / 100,
        assiduite_ordinal: parseFloat(document.getElementById('assiduite_ordinal').value) / 100,
        environnement_ordinal: parseFloat(document.getElementById('environnement_ordinal').value) / 100,
        sommeil_score: parseFloat(document.getElementById('sommeil_score').value) / 100,
        qualite_ordinal: parseFloat(document.getElementById('qualite_ordinal').value) / 100
    };
}

async function predict() {
    showLoader();
    
    const features = getFeatures();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(features)
        });
        
        const data = await response.json();
        
        if (data.success && data.data) {
            displayResults(data.data);
            showNotification('Analyse terminée avec succès !', 'success');
        } else {
            throw new Error(data.error || 'Erreur inconnue');
        }
    } catch (error) {
        console.error('Erreur:', error);
        showNotification(`Erreur: ${error.message}`, 'error');
    } finally {
        hideLoader();
    }
}

function displayResults(result) {
    // Update probability circle
    const circle = document.getElementById('probability-circle');
    const text = document.getElementById('probability-text');
    const probability = result.probability * 100;
    const circumference = 2 * Math.PI * 54;
    const offset = circumference - (probability / 100) * circumference;
    
    circle.style.strokeDashoffset = offset;
    text.textContent = `${probability.toFixed(1)}%`;
    
    // Update prediction title
    const title = document.getElementById('prediction-title');
    if (result.success) {
        title.textContent = '🎓 RÉUSSITE PRÉDITE';
        title.style.color = '#27ae60';
        circle.style.stroke = '#27ae60';
    } else {
        title.textContent = '⚠️ ÉCHEC PRÉDIT';
        title.style.color = '#e74c3c';
        circle.style.stroke = '#e74c3c';
    }
    
    // Update confidence
    document.getElementById('confidence-color').textContent = result.confidence_color;
    document.getElementById('confidence-text').textContent = `Confiance: ${result.confidence}`;
    document.getElementById('prediction-message').textContent = result.message;
    document.getElementById('recommendation-text').textContent = result.recommendation;
    document.getElementById('timestamp').textContent = new Date(result.timestamp).toLocaleString();
    
    // Display features analysis
    const featuresGrid = document.getElementById('features-grid');
    featuresGrid.innerHTML = '';
    
    result.features_analysis.forEach(feature => {
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        
        // Determine color based on value
        let color = '#e74c3c'; // red
        if (feature.value > 0.8) color = '#27ae60'; // green
        else if (feature.value > 0.6) color = '#3498db'; // blue
        else if (feature.value > 0.4) color = '#f39c12'; // orange
        
        featureItem.style.borderLeftColor = color;
        
        featureItem.innerHTML = `
            <div class="feature-header">
                <span class="feature-name">${feature.name}</span>
                <span class="feature-value">${feature.value.toFixed(2)}</span>
            </div>
            <div class="feature-impact" style="background: ${color}20; color: ${color};">
                ${feature.color} ${feature.impact}
            </div>
        `;
        
        featuresGrid.appendChild(featureItem);
    });
    
    // Show results
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

function setPreset(type) {
    const presets = {
        good: [90, 85, 80, 95, 90, 85, 95],
        average: [60, 50, 40, 70, 60, 50, 60],
        poor: [30, 20, 10, 40, 30, 20, 25]
    };
    
    const values = presets[type];
    
    sliders.forEach((sliderId, index) => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(`${sliderId}_value`);
        
        slider.value = values[index];
        valueDisplay.textContent = (values[index] / 100).toFixed(2);
    });
    
    const presetNames = {
        good: 'Excellent',
        average: 'Moyen',
        poor: 'Difficultés'
    };
    
    showNotification(`Profil "${presetNames[type]}" appliqué !`, 'info');
}

function resetForm() {
    sliders.forEach(sliderId => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(`${sliderId}_value`);
        
        slider.value = 50;
        valueDisplay.textContent = '0.50';
    });
    
    resultsContainer.style.display = 'none';
    showNotification('Formulaire réinitialisé', 'info');
}

function shareResults() {
    const features = getFeatures();
    const text = `🎓 Analyse de réussite étudiante:\n` +
                 `Niveau: ${features.niveau_etude.toFixed(2)}\n` +
                 `Heures: ${features.heures_etude_ordinal.toFixed(2)}\n` +
                 `Planning: ${features.planning_ordinal.toFixed(2)}\n` +
                 `Assiduité: ${features.assiduite_ordinal.toFixed(2)}\n` +
                 `Environnement: ${features.environnement_ordinal.toFixed(2)}\n` +
                 `Sommeil: ${features.sommeil_score.toFixed(2)}\n` +
                 `Qualité: ${features.qualite_ordinal.toFixed(2)}`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Analyse de réussite étudiante',
            text: text,
            url: window.location.href
        });
    } else {
        navigator.clipboard.writeText(text).then(() => {
            showNotification('Résultats copiés dans le presse-papier !', 'success');
        });
    }
}

function exportResults() {
    const features = getFeatures();
    const data = {
        features: features,
        timestamp: new Date().toISOString(),
        model_version: "2.0.0",
        threshold: MODEL_THRESHOLD
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    
    a.href = url;
    a.download = `analyse_etudiant_${new Date().getTime()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showNotification('Analyse exportée en JSON !', 'success');
}

function showLoader() {
    loader.style.display = 'flex';
}

function hideLoader() {
    loader.style.display = 'none';
}

function showNotification(message, type = 'info') {
    notification.textContent = message;
    notification.className = 'notification';
    
    // Add type class
    if (type === 'success') {
        notification.style.borderLeft = '5px solid #27ae60';
        notification.style.background = 'linear-gradient(135deg, #f1f8e9, #e8f5e9)';
    } else if (type === 'error') {
        notification.style.borderLeft = '5px solid #e74c3c';
        notification.style.background = 'linear-gradient(135deg, #ffebee, #fce4ec)';
    } else {
        notification.style.borderLeft = '5px solid #3498db';
        notification.style.background = 'linear-gradient(135deg, #e3f2fd, #e1f5fe)';
    }
    
    notification.classList.add('show');
    
    // Auto hide
    setTimeout(() => {
        notification.classList.remove('show');
    }, 5000);
}

// Service worker pour PWA (optionnel)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js').then(() => {
            console.log('Service Worker enregistré');
        }).catch(err => {
            console.log('Service Worker erreur:', err);
        });
    });
}