/* ============================================
   BIS Standards Finder - 3D Frontend Logic
   Three.js scene + API integration
   ============================================ */

const API_BASE = "http://localhost:8000";

// ==========================================
//  THREE.JS 3D BACKGROUND SCENE
// ==========================================
let scene, camera, renderer, particles, geometries = [], clock;
let mouseX = 0, mouseY = 0;

function initThreeScene() {
    const canvas = document.getElementById('three-canvas');
    scene = new THREE.Scene();
    clock = new THREE.Clock();

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 30;

    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);

    // --- Floating Particle Field ---
    const particleCount = 600;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);

    const palette = [
        new THREE.Color(0x60a5fa),  // blue
        new THREE.Color(0xa78bfa),  // purple
        new THREE.Color(0x22d3ee),  // cyan
        new THREE.Color(0x818cf8),  // indigo
    ];

    for (let i = 0; i < particleCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 80;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 60;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 50;

        const col = palette[Math.floor(Math.random() * palette.length)];
        colors[i * 3] = col.r;
        colors[i * 3 + 1] = col.g;
        colors[i * 3 + 2] = col.b;

        sizes[i] = Math.random() * 2.5 + 0.5;
    }

    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    particleGeo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const particleMat = new THREE.PointsMaterial({
        size: 1.5,
        vertexColors: true,
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
    });

    particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    // --- Floating geometric wireframes ---
    const wireframeMat = (color, opacity = 0.12) => new THREE.MeshBasicMaterial({
        color, wireframe: true, transparent: true, opacity,
    });

    const shapes = [
        { geo: new THREE.IcosahedronGeometry(3, 1), pos: [-20, 10, -10], color: 0x60a5fa, speed: 0.3 },
        { geo: new THREE.OctahedronGeometry(2.5, 0), pos: [22, -8, -15], color: 0xa78bfa, speed: 0.4 },
        { geo: new THREE.TorusGeometry(2, 0.6, 8, 16), pos: [-15, -12, -8], color: 0x22d3ee, speed: 0.25 },
        { geo: new THREE.TetrahedronGeometry(2, 0), pos: [18, 14, -12], color: 0x818cf8, speed: 0.35 },
        { geo: new THREE.DodecahedronGeometry(2, 0), pos: [0, -18, -20], color: 0xf472b6, speed: 0.2 },
        { geo: new THREE.BoxGeometry(3, 3, 3), pos: [-25, 0, -18], color: 0x34d399, speed: 0.15 },
    ];

    shapes.forEach(s => {
        const mesh = new THREE.Mesh(s.geo, wireframeMat(s.color));
        mesh.position.set(...s.pos);
        mesh.userData = { speed: s.speed, baseY: s.pos[1] };
        scene.add(mesh);
        geometries.push(mesh);
    });

    // --- Connection Lines ---
    const lineMat = new THREE.LineBasicMaterial({ color: 0x4338ca, transparent: true, opacity: 0.06 });
    for (let i = 0; i < 15; i++) {
        const points = [
            new THREE.Vector3((Math.random()-0.5)*60, (Math.random()-0.5)*40, (Math.random()-0.5)*30),
            new THREE.Vector3((Math.random()-0.5)*60, (Math.random()-0.5)*40, (Math.random()-0.5)*30),
        ];
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
        scene.add(new THREE.Line(lineGeo, lineMat));
    }

    // --- Mouse tracking ---
    document.addEventListener('mousemove', (e) => {
        mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    const t = clock.getElapsedTime();

    // Rotate particles slowly
    if (particles) {
        particles.rotation.y = t * 0.03;
        particles.rotation.x = t * 0.01;
    }

    // Animate geometric shapes
    geometries.forEach((mesh, i) => {
        mesh.rotation.x = t * mesh.userData.speed;
        mesh.rotation.y = t * mesh.userData.speed * 0.7;
        mesh.position.y = mesh.userData.baseY + Math.sin(t * mesh.userData.speed + i) * 2;
    });

    // Camera follows mouse subtly
    camera.position.x += (mouseX * 3 - camera.position.x) * 0.02;
    camera.position.y += (-mouseY * 2 - camera.position.y) * 0.02;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
}

// ==========================================
//  NAVIGATION
// ==========================================
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const section = link.dataset.section;
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');

        // Toggle sections
        document.getElementById('hero').style.display = section === 'search' ? 'block' : 'none';
        document.getElementById('search').style.display = section === 'search' ? 'block' : 'none';
        document.getElementById('results').style.display = section === 'search' && document.getElementById('results-grid').children.length > 0 ? 'block' : 'none';
        document.getElementById('browse').style.display = section === 'browse' ? 'block' : 'none';
        document.getElementById('about').style.display = section === 'about' ? 'block' : 'none';

        if (section === 'browse') loadStandards();
        if (section === 'about') loadAboutStats();
    });
});

// ==========================================
//  SEARCH FUNCTIONALITY
// ==========================================
function setQuery(el) {
    document.getElementById('query-input').value = el.textContent;
}

// Enter key submits
document.getElementById('query-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        performSearch();
    }
});

async function performSearch() {
    const query = document.getElementById('query-input').value.trim();
    if (!query) return;

    const topK = parseInt(document.getElementById('top-k').value);
    const category = document.getElementById('category-filter').value || null;
    const btn = document.getElementById('search-btn');

    // Show loading
    btn.disabled = true;
    document.getElementById('results').style.display = 'none';
    document.getElementById('loading').style.display = 'block';

    // Animate loading steps
    animateLoadingSteps();

    try {
        const response = await fetch(`${API_BASE}/api/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                product_description: query,
                top_k: topK,
                category_filter: category,
            }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (err) {
        console.error('Search failed:', err);
        alert(`Search failed: ${err.message}\n\nMake sure the FastAPI server is running on port 8000.`);
    } finally {
        btn.disabled = false;
        document.getElementById('loading').style.display = 'none';
    }
}

function animateLoadingSteps() {
    const steps = ['step-1', 'step-2', 'step-3', 'step-4'];
    steps.forEach(id => {
        document.getElementById(id).classList.remove('active', 'done');
    });
    document.getElementById('step-1').classList.add('active');

    setTimeout(() => {
        document.getElementById('step-1').classList.replace('active', 'done');
        document.getElementById('step-2').classList.add('active');
    }, 300);
    setTimeout(() => {
        document.getElementById('step-2').classList.replace('active', 'done');
        document.getElementById('step-3').classList.add('active');
    }, 600);
    setTimeout(() => {
        document.getElementById('step-3').classList.replace('active', 'done');
        document.getElementById('step-4').classList.add('active');
    }, 900);
}

// ==========================================
//  DISPLAY RESULTS
// ==========================================
function displayResults(data) {
    const { recommendations, latency_ms, source_chunks_count, method, source_documents } = data;

    // Metrics
    const metricsRow = document.getElementById('metrics-row');
    metricsRow.innerHTML = `
        <div class="metric-card metric-blue" style="animation-delay:0.1s">
            <div class="metric-value">${recommendations.length}</div>
            <div class="metric-label">Standards Found</div>
        </div>
        <div class="metric-card metric-green" style="animation-delay:0.2s">
            <div class="metric-value">${latency_ms.toFixed(0)}<small style="font-size:0.6em">ms</small></div>
            <div class="metric-label">Response Time</div>
        </div>
        <div class="metric-card metric-purple" style="animation-delay:0.3s">
            <div class="metric-value">${source_chunks_count}</div>
            <div class="metric-label">Chunks Searched</div>
        </div>
        <div class="metric-card metric-cyan" style="animation-delay:0.4s">
            <div class="metric-value">${method === 'gemini-rag' ? 'Gemini' : 'Local'}</div>
            <div class="metric-label">Engine Mode</div>
        </div>
    `;

    // Recommendation Cards
    const grid = document.getElementById('results-grid');
    grid.innerHTML = recommendations.map((rec, i) => `
        <div class="result-card rank-${rec.rank}" style="animation-delay:${0.15 * (i+1)}s">
            <div class="result-header">
                <span class="result-rank">#${rec.rank}</span>
                <span class="confidence-badge confidence-${rec.confidence.toLowerCase()}">${rec.confidence}</span>
            </div>
            <div class="result-std-id">${rec.standard_id}</div>
            <div class="result-title">${rec.title}</div>
            <div class="result-rationale">${rec.rationale}</div>
            ${rec.key_clauses && rec.key_clauses.length ? `
                <div class="result-clauses">
                    ${rec.key_clauses.map(c => `<div class="clause-tag">${escapeHtml(c)}</div>`).join('')}
                </div>
            ` : ''}
        </div>
    `).join('');

    // Source documents
    if (source_documents && source_documents.length) {
        const panel = document.getElementById('sources-panel');
        panel.style.display = 'block';
        document.getElementById('sources-content').innerHTML = source_documents.map(s => `
            <div class="source-item">
                <span class="source-id">${s.standard_id}</span> — ${s.title}
                <br><span class="source-score">Score: ${s.score}</span> | Category: ${s.category}
                <br><small style="color:var(--text-muted)">${escapeHtml(s.text_preview)}</small>
            </div>
        `).join('');
    }

    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function toggleSources() {
    const content = document.getElementById('sources-content');
    content.classList.toggle('open');
    const arrow = document.querySelector('.toggle-arrow');
    arrow.style.transform = content.classList.contains('open') ? 'rotate(180deg)' : '';
}

// ==========================================
//  BROWSE STANDARDS
// ==========================================
async function loadStandards(category = null) {
    try {
        const url = category ? `${API_BASE}/api/standards?category=${category}` : `${API_BASE}/api/standards`;
        const res = await fetch(url);
        const data = await res.json();

        // Category tabs
        const tabs = document.getElementById('category-tabs');
        const categories = ['All', ...(data.categories || [])];
        tabs.innerHTML = categories.map(c => `
            <button class="cat-tab ${(!category && c === 'All') || category === c ? 'active' : ''}" 
                    onclick="loadStandards(${c === 'All' ? 'null' : `'${c}'`})">
                ${c} ${c !== 'All' ? `(${data.standards.filter(s => s.category === c).length})` : `(${data.total})`}
            </button>
        `).join('');

        // Standards grid
        const grid = document.getElementById('standards-grid');
        grid.innerHTML = data.standards.map((s, i) => `
            <div class="standard-card" style="animation-delay:${i * 0.05}s">
                <div class="sc-id">${s.standard_id}</div>
                <div class="sc-title">${s.title}</div>
                <span class="sc-cat ${s.category}">${s.category}</span>
            </div>
        `).join('');
    } catch (err) {
        console.error('Failed to load standards:', err);
    }
}

// ==========================================
//  ABOUT STATS
// ==========================================
async function loadAboutStats() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        document.getElementById('about-stats').innerHTML = `
            <div class="about-stat"><div class="stat-val">${data.standards_count}</div><div class="stat-label">BIS Standards</div></div>
            <div class="about-stat"><div class="stat-val">${data.chunks_count}</div><div class="stat-label">Document Chunks</div></div>
            <div class="about-stat"><div class="stat-val">${data.llm_mode === 'gemini' ? 'Gemini' : 'Local'}</div><div class="stat-label">LLM Mode</div></div>
        `;
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
}

// ==========================================
//  HEALTH CHECK
// ==========================================
async function checkHealth() {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        dot.className = 'status-dot online';
        text.textContent = `Engine Online (${data.llm_mode})`;
    } catch {
        dot.className = 'status-dot error';
        text.textContent = 'Engine Offline';
    }
}

// ==========================================
//  UTILITIES
// ==========================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==========================================
//  INIT
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    initThreeScene();
    checkHealth();
    setInterval(checkHealth, 15000);
});
