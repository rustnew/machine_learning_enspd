// Configuration des caractéristiques
const featuresConfig = [
    {
        id: 'niveau_etude',
        name: 'Niveau d\'Étude',
        icon: 'fas fa-graduation-cap',
        description: 'Niveau académique actuel',
        min: 0,
        max: 1,
        step: 0.01,
        value: 0.7
    },
    {
        id: 'heures_etude_ordinal',
        name: 'Heures d\'Étude',
        icon: 'fas fa-clock',
        description: 'Heures par semaine dédiées à l\'étude',
        min: 0,
        max: 1,
        step: 0.01,
        value: 0.6
    },
    {
        id: 'planning_ordinal',
        name: 'Planification',
        icon: 'fas fa-calendar-alt',
        description: 'Organisation des sessions d\'étude',
        min: 0,
        max: 1,
        step: 0.01,
        value: 0.5
    },
    {
        id: 'assiduite_ordinal',
        name: 'Assiduité',
        icon: 'fas fa-user-check',
        description: 'Taux de présence aux cours',
        min: 0,
        max: 1,
        step: 0.01,
        value: 0.8
    },
    {
        id: 'environnement_ordinal',
        name: 'Environnement',
        icon: 'fas fa-home',
        description: 'Qualité de l\'environnement d\'étude',
        min: 0,
        max: 1,
        step: 0.01,
        value: 0.9
    },
    {
        id: 'sommeil_score',
        name: 'Qualité de Sommeil',
        icon: 'fas fa-bed',
        description: 'Durée et qualité du repos quotidien',
        min: 0,
        max: 1,
        step: 0.01,
        value: 0.7
    },
    {
        id: 'qualite_ordinal',
        name: 'Qualité d\'Étude',
        icon: 'fas fa-star',
        description: 'Concentration et efficacité',
        min: 0,
        max: 1,
        step: 0.01,
        value: 0.6
    }
];

// Initialisation de l'interface
function initInterface() {
    const container = document.getElementById('featuresContainer');
    
    featuresConfig.forEach(feature => {
        const sliderHTML = `
            <div class="feature-slider">
                <div class="feature-header">
                    <div class="feature-name">
                        <i class="${feature.icon}"></i>
                        ${feature.name}
                    </div>
                    <div class="feature-value" id="${feature.id}-value">
                        ${feature.value.toFixed(2)}
                    </div>
                </div>
                
                <div class="slider-container">
                    <input type="range" 
                           id="${feature.id}" 
                           min="${feature.min}" 
                           max="${feature.max}" 
                           step="${feature.step}" 
                           value="${feature.value}"
                           oninput="updateSliderValue('${feature.id}', this.value)"
                           class="feature-slider-input">
                    <div class="slider-labels">
                        <span>Faible</span>
                        <span>Moyen</span>
                        <span>Fort</span>
                    </div>
                </div>
                
                <div class="feature-description">
                    ${feature.description}
                </div>
            </div>
        `;
        
        container.innerHTML += sliderHTML;
    });
}

// Mise à jour des valeurs des curseurs
function updateSliderValue(id, value) {
    const valueElement = document.getElementById(`${id}-value`);
    const floatValue = parseFloat(value);
    valueElement.textContent = floatValue.toFixed(2);
    
    valueElement.classList.add('updated');
    setTimeout(() => {
        valueElement.classList.remove('updated');
    }, 300);
}

// Génération d'un profil aléatoire
function generateRandom() {
    featuresConfig.forEach(feature => {
        const randomValue = Math.max(0, Math.min(1, 
            (Math.random() * 0.4) + 0.4
        )).toFixed(2);
        
        const slider = document.getElementById(feature.id);
        const valueElement = document.getElementById(`${feature.id}-value`);
        
        slider.value = randomValue;
        valueElement.textContent = randomValue;
    });
    
    showNotification('Profil aléatoire généré', 'info');
}

// Réinitialisation du formulaire
function resetForm() {
    featuresConfig.forEach(feature => {
        const slider = document.getElementById(feature.id);
        const valueElement = document.getElementById(`${feature.id}-value`);
        
        slider.value = feature.value;
        valueElement.textContent = feature.value.toFixed(2);
    });
    
    document.getElementById('resultsPlaceholder').style.display = 'block';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('detailedResults').innerHTML = '';
    
    showNotification('Formulaire réinitialisé', 'info');
}

// Affichage des notifications
function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: ${type === 'success' ? '#10b981' : '#0a3d2e'};
        color: white;
        border-radius: 8px;
        font-weight: 500;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Gestion du chargement
function showLoading() {
    document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

// Analyse principale
async function analyzeStudent() {
    const data = {};
    let isValid = true;
    
    featuresConfig.forEach(feature => {
        const value = parseFloat(document.getElementById(feature.id).value);
        data[feature.id] = value;
        
        if (isNaN(value) || value < 0 || value > 1) {
            isValid = false;
            const element = document.getElementById(feature.id);
            element.style.borderColor = '#dc2626';
            setTimeout(() => {
                element.style.borderColor = '';
            }, 2000);
        }
    });
    
    if (!isValid) {
        showNotification('Valeurs invalides détectées', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const result = await simulateAPICall(data);
        displayResults(result);
        showNotification('Analyse terminée avec succès', 'success');
    } catch (error) {
        showNotification('Erreur lors de l\'analyse', 'error');
        console.error('Erreur:', error);
    } finally {
        hideLoading();
    }
}

// Simulation d'appel API
function simulateAPICall(data) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const totalScore = Object.values(data).reduce((sum, val) => sum + val, 0);
            const averageScore = totalScore / Object.keys(data).length;
            const probability = averageScore * 0.9 + Math.random() * 0.1;
            
            const result = {
                probability: Math.min(0.99, Math.max(0.01, probability)),
                success: probability > 0.6,
                confidence: probability > 0.8 ? 'Élevée' : probability > 0.6 ? 'Moyenne' : 'Faible',
                timestamp: new Date().toISOString(),
                features_analysis: featuresConfig.map(feature => ({
                    name: feature.name,
                    value: data[feature.id],
                    impact: data[feature.id] > 0.7 ? 'Positif' : data[feature.id] > 0.4 ? 'Neutre' : 'À améliorer',
                    color: data[feature.id] > 0.7 ? '#10b981' : data[feature.id] > 0.4 ? '#f59e0b' : '#dc2626'
                }))
            };
            
            resolve(result);
        }, 1500);
    });
}

// Affichage des résultats
function displayResults(result) {
    document.getElementById('resultsPlaceholder').style.display = 'none';
    
    const resultsHTML = `
        <div class="probability-display">
            <div class="probability-circle" style="--progress: ${result.probability * 100}%">
                <div class="probability-value">${(result.probability * 100).toFixed(1)}%</div>
            </div>
            
            <div class="result-status ${result.success ? 'success' : 'failure'}">
                ${result.success ? '✓ RÉUSSITE PRÉDITE' : '⚠ RISQUE D\'ÉCHEC'}
                <span style="margin-left: 8px; font-size: 0.875rem;">
                    (Confiance: ${result.confidence})
                </span>
            </div>
        </div>
        
        <div style="margin-top: 24px; padding: 20px; background: #f8fafc; border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <i class="fas fa-lightbulb" style="color: #f59e0b;"></i>
                <h3 style="font-size: 1rem; color: #1a1a1a;">
                    ${result.success ? 'Recommandations d\'optimisation' : 'Actions prioritaires'}
                </h3>
            </div>
            <p style="color: #64748b; line-height: 1.6;">
                ${result.success ? 
                    'Le profil présente des caractéristiques favorables à la réussite. Maintenez ces bonnes pratiques.' :
                    'Des améliorations sont nécessaires dans plusieurs domaines. Concentrez-vous sur les points faibles identifiés.'
                }
            </p>
        </div>
    `;
    
    const container = document.getElementById('resultsContainer');
    container.innerHTML = resultsHTML;
    container.style.display = 'block';
    
    displayFeatureAnalysis(result.features_analysis);
}

// Analyse détaillée des caractéristiques
function displayFeatureAnalysis(features) {
    let analysisHTML = `
        <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <i class="fas fa-chart-bar" style="color: #0a3d2e;"></i>
                <h3 style="font-size: 1.125rem; color: #1a1a1a;">
                    Analyse par Caractéristique
                </h3>
            </div>
        </div>
        
        <div class="analysis-grid">
    `;
    
    features.forEach(feature => {
        const percentage = feature.value * 100;
        
        analysisHTML += `
            <div class="feature-card">
                <div class="feature-card-title">
                    <span>${feature.name}</span>
                    <span style="color: #0a3d2e; font-weight: 600;">
                        ${percentage.toFixed(0)}%
                    </span>
                </div>
                
                <div class="feature-bar">
                    <div class="feature-fill" style="width: ${percentage}%"></div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                    <span style="display: flex; align-items: center; gap: 4px; color: ${feature.color}; font-size: 0.875rem;">
                        <i class="fas fa-${feature.impact === 'Positif' ? 'check' : feature.impact === 'Neutre' ? 'minus' : 'exclamation'}"></i>
                        ${feature.impact}
                    </span>
                    <span style="font-size: 0.875rem; color: #64748b; font-weight: 500;">
                        ${feature.value.toFixed(2)}
                    </span>
                </div>
            </div>
        `;
    });
    
    analysisHTML += '</div>';
    document.getElementById('detailedResults').innerHTML = analysisHTML;
}

// Initialisation
window.onload = function() {
    initInterface();
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        .feature-value.updated {
            animation: pulse 0.3s ease;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    `;
    document.head.appendChild(style);
};