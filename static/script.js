document.getElementById('recommendForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const userId = document.getElementById('userId').value.trim();
    const textQuery = document.getElementById('textQuery').value.trim();
    const imageFile = document.getElementById('imageUpload').files[0];
    const k = document.getElementById('kResults').value;
    
    const statusMsg = document.getElementById('statusMessage');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const resultsGrid = document.getElementById('resultsGrid');
    const latencyBadge = document.getElementById('latencyBadge');
    
    // Reset state
    statusMsg.textContent = '';
    statusMsg.className = 'status-message';
    
    if (!userId && !textQuery && !imageFile) {
        statusMsg.textContent = 'Please provide a User ID, text query, or an image.';
        statusMsg.classList.add('error');
        return;
    }
    
    submitBtn.disabled = true;
    btnText.textContent = 'Processing...';
    btnSpinner.classList.remove('hidden');
    
    // Show skeleton loading state
    showSkeletonLoading();
    
    const formData = new FormData();
    if (userId) formData.append('user_id', userId);
    if (textQuery) formData.append('query', textQuery);
    if (imageFile) formData.append('image', imageFile);
    formData.append('k', k);
    
    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        renderResults(data);
        
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        statusMsg.textContent = 'Failed to fetch recommendations. Check the console for details.';
        statusMsg.classList.add('error');
        resultsGrid.innerHTML = '<div class="empty-state"><p>Something went wrong. Please try again.</p></div>';
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = 'Get Recommendations';
        btnSpinner.classList.add('hidden');
    }
});

function renderResults(data) {
    const resultsGrid = document.getElementById('resultsGrid');
    const latencyBadge = document.getElementById('latencyBadge');
    
    resultsGrid.innerHTML = '';
    
    if (!data.recommendations || data.recommendations.length === 0) {
        resultsGrid.innerHTML = '<div class="empty-state"><p>No recommendations found for this query.</p></div>';
        latencyBadge.classList.add('hidden');
        return;
    }
    
    // Update latency badge
    latencyBadge.textContent = `${data.retrieval_latency_ms} ms`;
    latencyBadge.classList.remove('hidden');
    
    // Min-max rescale cosine similarities to a user-friendly 75-99% range.
    // Raw cosine sim in the Two-Tower latent space is typically 0.2-0.5,
    // which is normal but looks bad as a raw %. Production systems always rescale.
    const scores = data.recommendations
        .map(item => item.cosine_sim)
        .filter(s => s !== undefined);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);
    const scoreRange = maxScore - minScore || 1; // avoid division by zero

    // Render cards with staggered entrance
    data.recommendations.forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'product-card';
        card.style.animationDelay = `${index * 0.04}s`;
        
        // Format price
        const priceStr = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(item.price);
        
        // Rescale to 75-99% display range
        let matchBadge = '';
        if (item.cosine_sim !== undefined) {
            const normalized = (item.cosine_sim - minScore) / scoreRange; // 0 to 1
            const displayPct = Math.round(75 + normalized * 24);         // 75 to 99
            matchBadge = `<span class="product-match">${displayPct}% match</span>`;
        }
        
        // Build image URL from the local path (data/images/12345.jpg → /images/12345.jpg)
        let imgHtml = '';
        if (item.image_url) {
            const filename = item.image_url.split('/').pop().split('\\').pop();
            imgHtml = `<img class="product-img" src="/images/${filename}" alt="${escapeHTML(item.title)}" loading="lazy" onerror="this.style.display='none'">`;
        }
        
        card.innerHTML = `
            ${imgHtml}
            <div class="product-body">
                <div class="product-category">${escapeHTML(item.category)}</div>
                <div class="product-title">${escapeHTML(item.title)}</div>
                <div class="product-footer">
                    <span class="product-price">${priceStr}</span>
                    ${matchBadge}
                </div>
            </div>
        `;
        
        resultsGrid.appendChild(card);
    });
}

function escapeHTML(str) {
    if (!str) return '';
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

// ── Image Preview Logic ────────────────
const imageUpload = document.getElementById('imageUpload');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');
const imagePreview = document.getElementById('imagePreview');
const clearImageBtn = document.getElementById('clearImageBtn');
const fileUploadLabel = document.getElementById('fileUploadLabel');

imageUpload.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreviewContainer.classList.remove('hidden');
            fileUploadLabel.style.display = 'none';
        }
        reader.readAsDataURL(file);
    } else {
        clearImage();
    }
});

clearImageBtn.addEventListener('click', clearImage);

function clearImage() {
    imageUpload.value = '';
    imagePreview.src = '';
    imagePreviewContainer.classList.add('hidden');
    fileUploadLabel.style.display = '';
}

// ── Drag & Drop ────────────────────────
const dropZone = document.getElementById('dropZone');

['dragenter', 'dragover'].forEach(evtName => {
    dropZone.addEventListener(evtName, (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent-color)';
        dropZone.style.background = 'rgba(59,130,246,0.04)';
    });
});

['dragleave', 'drop'].forEach(evtName => {
    dropZone.addEventListener(evtName, (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '';
        dropZone.style.background = '';
    });
});

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageUpload.files = files;
        imageUpload.dispatchEvent(new Event('change'));
    }
});

// ── Skeleton Loading ───────────────────
function showSkeletonLoading() {
    const resultsGrid = document.getElementById('resultsGrid');
    const latencyBadge = document.getElementById('latencyBadge');
    
    latencyBadge.classList.add('hidden');
    resultsGrid.innerHTML = '';
    
    for (let i = 0; i < 8; i++) {
        const skeleton = document.createElement('div');
        skeleton.className = 'skeleton-card';
        skeleton.innerHTML = `
            <div class="skeleton skeleton-text short"></div>
            <div>
                <div class="skeleton skeleton-title"></div>
                <div class="skeleton skeleton-title" style="width: 55%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: auto; padding-top: 0.75rem; border-top: 1px solid #f1f5f9;">
                <div class="skeleton skeleton-text short"></div>
                <div class="skeleton skeleton-text" style="width: 25%"></div>
            </div>
        `;
        resultsGrid.appendChild(skeleton);
    }
}
