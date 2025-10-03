// Frontend main script to wire the static UI to the FastAPI backend

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const loading = document.getElementById('loading');
const uploadSection = document.getElementById('uploadSection');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const successMessage = document.getElementById('successMessage');

function setStatusOnline() {
    statusDot.classList.remove('offline');
    statusDot.classList.add('online');
    statusText.textContent = 'API Online';
}

function setStatusOffline() {
    statusDot.classList.remove('online');
    statusDot.classList.add('offline');
    statusText.textContent = 'API Offline';
}

async function checkApiStatus() {
    try {
        const res = await fetch('/status');
        if (res.ok) {
            setStatusOnline();
        } else {
            setStatusOffline();
        }
    } catch (err) {
        setStatusOffline();
    }
}

// Run once at load and poll every 10s
checkApiStatus();
setInterval(checkApiStatus, 10000);

// Hook upload button to file input
uploadButton.addEventListener('click', () => fileInput.click());

// Reset UI to initial upload state
function resetUpload() {
    errorMessage.textContent = '';
    successMessage.textContent = '';
    resultsSection.style.display = 'none';
    loading.style.display = 'none';
    uploadSection.style.display = 'block';
    fileInput.value = '';
    document.getElementById('madeShots').textContent = '0';
    document.getElementById('missedShots').textContent = '0';
    document.getElementById('totalAttempts').textContent = '0';
    document.getElementById('accuracy').textContent = '0%';
}

// Attach to global for HTML button onclick
window.resetUpload = resetUpload;

// Handle file selection and upload
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    // Simple client-side validation
    uploadSection.style.display = 'none';
    loading.style.display = 'block';
    errorMessage.textContent = '';
    successMessage.textContent = '';

    const formData = new FormData();
    formData.append('video', file, file.name);

    try {
        const resp = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            let text;
            try { text = await resp.text(); } catch { text = resp.statusText; }
            throw new Error(`Server responded ${resp.status}: ${text}`);
        }

        const json = await resp.json();
        document.getElementById('madeShots').textContent = json.made_shots ?? 0;
        document.getElementById('missedShots').textContent = json.missed_shots ?? 0;
        document.getElementById('totalAttempts').textContent = json.total_attempts ?? 0;
        document.getElementById('accuracy').textContent = (json.accuracy ?? 0) + '%';

        loading.style.display = 'none';
        resultsSection.style.display = 'block';
        successMessage.textContent = 'Analysis complete';
    } catch (err) {
        loading.style.display = 'none';
        errorMessage.textContent = 'Analysis failed: ' + (err.message || err);
        uploadSection.style.display = 'block';
    }
});

// Tab switching for UI
function switchTab(tab) {
    const analyzeTab = document.getElementById('analyzeTab');
    const trainTab = document.getElementById('trainTab');
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(t => t.classList.remove('active'));
    if (tab === 'analyze') {
        analyzeTab.classList.add('active');
        trainTab.classList.remove('active');
        document.querySelector('button[onclick="switchTab(\'analyze\')"]').classList.add('active');
    } else {
        analyzeTab.classList.remove('active');
        trainTab.classList.add('active');
        document.querySelector('button[onclick="switchTab(\'train\')"]').classList.add('active');
    }
}
window.switchTab = switchTab;

// Minimal stubs for training UI referenced in HTML (disabled by default)
window.handleDatasetFiles = function (e) {
    const list = document.getElementById('fileList');
    list.innerHTML = '';
    Array.from(e.target.files).forEach(f => {
        const el = document.createElement('div');
        el.textContent = f.name;
        list.appendChild(el);
    });
};
window.startTraining = function () {
    alert('Training endpoint is not implemented in this deployment. Use the UI to upload datasets to your training service.');
};