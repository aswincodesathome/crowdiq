/**
 * CrowdIQ Frontend — Backend API Integration
 * Real-time updates from Flask backend
 */

// Smart API URL detection
let API_BASE = '';
function initializeAPIBase() {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    
    // Use current host if available, fallback to localhost
    if (hostname && hostname !== '' && hostname !== '127.0.0.1') {
        API_BASE = `${protocol}//${hostname}:5000/api`;
    } else {
        API_BASE = `${protocol}//localhost:5000/api`;
    }
    
    console.log('API Base URL:', API_BASE);
}

// Initialize on load
initializeAPIBase();

let LIVE_UPDATES_INTERVAL = null;

/**
 * Fetch current predictions from backend
 */
async function fetchLivePredictions(city = 'Chennai') {
    try {
        const url = `${API_BASE}/predictions/${city}`;
        console.log(`[${new Date().toLocaleTimeString()}] Fetching: ${url}`);
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            console.error(`❌ HTTP ${response.status} from ${url}`);
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        console.log(`✓ Got predictions for ${city}:`, data);
        return data;
    } catch (error) {
        console.error(`❌ Error fetching predictions for ${city}:`, error.message);
        return null;
    }
}

/**
 * Fetch 24-hour forecast
 */
async function fetchForecast(city = 'Chennai') {
    try {
        const url = `${API_BASE}/forecast/${city}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error(`❌ Error fetching forecast for ${city}:`, error.message);
        return [];
    }
}

/**
 * Fetch available cities and their model status
 */
async function fetchCities() {
    try {
        const response = await fetch(`${API_BASE}/cities`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching cities:', error);
        return [];
    }
}

/**
 * Get backend health status with detailed diagnostics
 */
async function checkBackendHealth() {
    try {
        const url = `${API_BASE}/health`;
        console.log(`[HEALTH CHECK] Attempting: ${url}`);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            },
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            console.error(`❌ Backend returned HTTP ${response.status}`);
            return { status: 'offline', reason: `HTTP ${response.status}` };
        }
        
        const data = await response.json();
        console.log('✅ Backend health check passed:', data);
        return data;
    } catch (error) {
        console.error('❌ Backend health check failed:', error);
        console.error('Error type:', error.name);
        console.error('Error message:', error.message);
        
        if (error.name === 'AbortError') {
            console.error('Request timed out after 5 seconds');
        }
        
        return { 
            status: 'offline',
            reason: error.message,
            type: error.name 
        };
    }
}

/**
 * Update dashboard with live data
 */
async function updateDashboardLive(city = 'Chennai') {
    const data = await fetchLivePredictions(city);
    if (!data) {
        console.log('No live data available');
        return;
    }

    // Update stats
    if (data.overall_avg !== undefined) {
        const avgEl = document.getElementById('s-crowd');
        if (avgEl) avgEl.textContent = Math.round(data.overall_avg) + '%';
    }

    if (data.routes) {
        const routeCount = Object.keys(data.routes).length;
        const routesEl = document.getElementById('s-routes');
        if (routesEl) routesEl.textContent = routeCount;
    }

    // Update predictions if available
    if (data.predictions && data.predictions.xgboost) {
        const xgb = data.predictions.xgboost;
        const peakEl = document.getElementById('s-delay');
        if (peakEl) peakEl.textContent = xgb.peak_hour + ':00';
    }

    // Update route details
    if (data.routes) {
        updateRouteDetails(data.routes);
    }

    // Update timestamp
    const updated = new Date(data.timestamp);
    const lastUpdateEl = document.getElementById('last-update');
    if (lastUpdateEl) {
        lastUpdateEl.textContent = `Updated: ${updated.toLocaleTimeString('en-IN', {hour: '2-digit', minute: '2-digit'})}`;
    }
}

/**
 * Update route details panel
 */
function updateRouteDetails(routes) {
    const routeListEl = document.getElementById('route-list');
    if (!routeListEl) return;

    let html = '';
    for (const [routeId, stats] of Object.entries(routes)) {
        const crowd = stats.current || stats.avg_crowd || 0;
        const crowdPct = Math.min(100, crowd);
        const color = crowdPct < 40 ? '#34d399' : crowdPct < 70 ? '#fbbf24' : '#f87171';

        html += `
            <div class="route-item" style="border-left: 3px solid ${color}">
                <div class="route-badge" style="background: ${color}20; color: ${color}">${routeId}</div>
                <div class="route-info">
                    <div class="route-name">Route ${routeId}</div>
                    <div class="route-detail">Now: ${Math.round(crowdPct)}% · Peak: ${Math.round(stats.max_crowd || 0)}%</div>
                </div>
                <div class="crowd-bar" style="background: #1a2235">
                    <div class="crowd-fill" style="width: ${crowdPct}%; background: ${color}"></div>
                </div>
            </div>
        `;
    }

    routeListEl.innerHTML = html;
}

/**
 * Update forecast chart with live data
 */
async function updateForecastChart(city = 'Chennai') {
    const forecast = await fetchForecast(city);
    if (!forecast || forecast.length === 0) return;

    const ctx = document.getElementById('forecastChart');
    if (!ctx) return;

    // Destroy existing chart
    if (window.forecastChart) {
        window.forecastChart.destroy();
    }

    const labels = forecast.map(f => {
        const hour = f.hour;
        return `${String(hour).padStart(2, '0')}:00`;
    });

    const crowds = forecast.map(f => f.crowd);

    window.forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Predicted Crowd %',
                data: crowds,
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 2,
                pointBackgroundColor: '#38bdf8'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b', font: { size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: {
                        color: '#64748b',
                        font: { size: 10 },
                        callback: v => v + '%'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

/**
 * Start real-time updates
 */
function startLiveUpdates(city = 'Chennai') {
    // Update immediately
    updateDashboardLive(city);
    updateForecastChart(city);

    // Update every 10 seconds
    if (LIVE_UPDATES_INTERVAL) {
        clearInterval(LIVE_UPDATES_INTERVAL);
    }

    LIVE_UPDATES_INTERVAL = setInterval(() => {
        updateDashboardLive(city);
    }, 10000);
}

/**
 * Stop live updates
 */
function stopLiveUpdates() {
    if (LIVE_UPDATES_INTERVAL) {
        clearInterval(LIVE_UPDATES_INTERVAL);
        LIVE_UPDATES_INTERVAL = null;
    }
}

/**
 * Initialize backend connection with detailed feedback
 */
async function initBackendConnection() {
    const statusEl = document.getElementById('backend-status');
    
    if (statusEl) {
        statusEl.innerHTML = '<div style="color: #94a3b8; font-size: 12px">🔄 Connecting...</div>';
    }
    
    const health = await checkBackendHealth();
    
    if (health.status === 'offline') {
        console.error('❌ Backend is offline!');
        const reason = health.reason || 'Unknown error';
        if (statusEl) {
            statusEl.innerHTML = `<div style="color: #f87171; font-size: 12px">❌ Backend offline<br/><span style="font-size: 10px">${reason}</span></div>`;
        }
        return false;
    } else {
        console.log('✅ Backend connected successfully');
        if (statusEl) {
            statusEl.innerHTML = '<div style="color: #34d399; font-size: 12px">✅ Live data active</div>';
        }
        return true;
    }
}

/**
 * Handle city switch with backend integration
 */
async function switchCityLive(city) {
    stopLiveUpdates();
    startLiveUpdates(city);
}

// Initialize on page load with retry logic
document.addEventListener('DOMContentLoaded', async () => {
    console.log('📄 Page loaded, initializing backend connection...');
    console.log('API Base URL:', API_BASE);
    
    let isConnected = false;
    let attempts = 0;
    const maxAttempts = 3;
    
    // Try to connect with retries
    while (!isConnected && attempts < maxAttempts) {
        attempts++;
        console.log(`Attempt ${attempts}/${maxAttempts}...`);
        
        isConnected = await initBackendConnection();
        
        if (!isConnected && attempts < maxAttempts) {
            console.log('Retrying in 2 seconds...');
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }
    
    if (isConnected) {
        console.log('✅ Connection successful, starting live updates');
        startLiveUpdates('Chennai');
    } else {
        console.error('❌ Failed to connect after 3 attempts');
        // Show error and offer manual retry
        const statusEl = document.getElementById('backend-status');
        if (statusEl) {
            statusEl.innerHTML = `<div style="color: #f87171; font-size: 12px">❌ No backend connection<br/><button onclick="location.reload()" style="margin-top: 4px; padding: 2px 8px; font-size: 11px">Retry</button></div>`;
        }
    }
    
    // Set up periodic health checks every 30 seconds
    setInterval(async () => {
        const health = await checkBackendHealth();
        if (health.status !== 'offline' && !LIVE_UPDATES_INTERVAL) {
            console.log('Backend came back online, restarting updates');
            startLiveUpdates('Chennai');
        }
    }, 30000);
});
