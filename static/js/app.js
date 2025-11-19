// Nail Disease Screening Application - Frontend JavaScript

const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const detectionVis = document.getElementById('detectionVis');
const detectionImage = document.getElementById('detectionImage');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;

// Upload area click handler
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// File input change handler
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    } else {
        showError('Please drop a valid image file.');
    }
});

// Handle file selection
function handleFileSelect(file) {
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size exceeds 16MB limit. Please choose a smaller image.');
        return;
    }

    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        analyzeBtn.disabled = false;
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

// Analyze button handler
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    // Show loading, hide results
    showLoading();
    hideResults();
    hideError();

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'An error occurred during analysis');
        }

        // Display results
        displayResults(data);
    } catch (error) {
        showError(error.message || 'Failed to analyze image. Please try again.');
    } finally {
        hideLoading();
    }
});

// Display results
function displayResults(data) {
    resultsSection.style.display = 'block';

    // Show detection visualization if available
    if (data.detection_visualization) {
        detectionImage.src = 'data:image/jpeg;base64,' + data.detection_visualization;
        detectionVis.style.display = 'block';
    } else {
        detectionVis.style.display = 'none';
    }

    // Clear previous results
    resultsContainer.innerHTML = '';

    // Display each nail result
    if (data.results && data.results.length > 0) {
        data.results.forEach((result, index) => {
            const resultCard = createResultCard(result, index);
            resultsContainer.appendChild(resultCard);
        });
    } else {
        resultsContainer.innerHTML = '<p>No results to display.</p>';
    }
}

// Create result card
function createResultCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const diseaseName = formatDiseaseName(result.disease);
    const probability = (result.probability * 100).toFixed(1);

    card.innerHTML = `
        <h3>
            <span class="nail-badge">Nail ${result.nail_index}</span>
            ${diseaseName}
        </h3>
        
        <div class="prediction">
            <div class="prediction-label">
                <span>Predicted Condition:</span>
                <span class="prediction-probability">${probability}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${probability}%"></div>
            </div>
        </div>

        <div class="all-probabilities">
            <h4>All Condition Probabilities:</h4>
            ${Object.entries(result.all_probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([disease, prob]) => `
                    <div class="probability-item">
                        <span class="probability-item-label">${formatDiseaseName(disease)}</span>
                        <span class="probability-item-value">${(prob * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
        </div>

        <div style="margin-top: 15px; font-size: 0.85em; color: #999;">
            Detection Confidence: ${(result.detection_confidence * 100).toFixed(1)}%
        </div>
    `;

    return card;
}

// Format disease name for display
function formatDiseaseName(disease) {
    return disease
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

// Show/hide functions
function showLoading() {
    loadingSection.style.display = 'block';
    analyzeBtn.disabled = true;
}

function hideLoading() {
    loadingSection.style.display = 'none';
    analyzeBtn.disabled = false;
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
}

function hideError() {
    errorSection.style.display = 'none';
}

