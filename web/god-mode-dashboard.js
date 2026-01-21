// HEAN God-Mode Dashboard: Market Omniscience Visualization

const API_BASE = '/api';

// State
let probabilityManifold = null;
let futureCone = null;
let updateInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('God-Mode Dashboard: Initializing Market Omniscience...');
    
    initializeProbabilityManifold();
    initializeFutureCone();
    startDataUpdates();
    updateSystemStatus();
    
    console.log('God-Mode Dashboard: Ready. Market omniscience achieved.');
});

// Probability Manifold: 3D visualization of market state space
function initializeProbabilityManifold() {
    const container = document.getElementById('probability-manifold');
    
    // Generate sample probability distribution (in production, fetch from API)
    const data = generateProbabilityManifold();
    
    const trace = {
        x: data.x,
        y: data.y,
        z: data.z,
        colorscale: [[0, '#ff4444'], [0.5, '#00d4ff'], [1, '#00ff88']],
        type: 'surface',
        showscale: true,
        colorbar: {
            title: 'Probability',
            titleside: 'right'
        },
        lighting: {
            ambient: 0.8,
            diffuse: 0.8,
            specular: 0.1,
            roughness: 0.9,
            fresnel: 0.1
        }
    };
    
    const layout = {
        title: {
            text: 'Market State Probability Distribution',
            font: { color: '#00d4ff', size: 16 }
        },
        scene: {
            xaxis: { title: 'Price Momentum', color: '#a0a0a0', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'Sentiment', color: '#a0a0a0', gridcolor: 'rgba(255,255,255,0.1)' },
            zaxis: { title: 'Probability', color: '#a0a0a0', gridcolor: 'rgba(255,255,255,0.1)' },
            bgcolor: 'rgba(0,0,0,0.5)',
            camera: {
                eye: { x: 1.5, y: 1.5, z: 1.5 }
            }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e0e0e0' },
        margin: { l: 0, r: 0, t: 50, b: 0 }
    };
    
    const config = {
        displayModeBar: false,
        responsive: true
    };
    
    Plotly.newPlot(container, [trace], layout, config);
    
    probabilityManifold = { container, trace, layout };
}

function generateProbabilityManifold() {
    // Generate 3D probability surface
    const size = 50;
    const x = [];
    const y = [];
    const z = [];
    
    for (let i = 0; i < size; i++) {
        x[i] = [];
        y[i] = [];
        z[i] = [];
        
        for (let j = 0; j < size; j++) {
            const priceMomentum = (i / size) * 2 - 1; // -1 to 1
            const sentiment = (j / size) * 2 - 1; // -1 to 1
            
            // Generate probability based on Gaussian-like distribution
            // Higher probability where momentum and sentiment align
            const distance = Math.sqrt(priceMomentum ** 2 + sentiment ** 2);
            const probability = Math.exp(-distance * 2) * (1 + Math.sin(distance * 5) * 0.3);
            
            x[i][j] = priceMomentum;
            y[i][j] = sentiment;
            z[i][j] = probability;
        }
    }
    
    return { x, y, z };
}

// Future Cone: Visualization of where the system is pushing the market
function initializeFutureCone() {
    const container = document.getElementById('future-cone');
    
    // Generate future cone data
    const data = generateFutureCone();
    
    const traces = [];
    
    // Main cone trajectory (bright)
    traces.push({
        x: data.trajectory.time,
        y: data.trajectory.price,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Active Market Push',
        line: {
            color: '#00ff88',
            width: 4
        },
        marker: {
            size: 8,
            color: '#00ff88'
        }
    });
    
    // Upper uncertainty bound
    traces.push({
        x: data.upper.time,
        y: data.upper.price,
        mode: 'lines',
        type: 'scatter',
        name: 'Upper Bound',
        line: {
            color: 'rgba(0, 212, 255, 0.3)',
            width: 2,
            dash: 'dash'
        },
        fill: 'tonexty',
        fillcolor: 'rgba(0, 212, 255, 0.1)'
    });
    
    // Lower uncertainty bound
    traces.push({
        x: data.lower.time,
        y: data.lower.price,
        mode: 'lines',
        type: 'scatter',
        name: 'Lower Bound',
        line: {
            color: 'rgba(0, 212, 255, 0.3)',
            width: 2,
            dash: 'dash'
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(0, 212, 255, 0.1)'
    });
    
    // High-probability profit zones
    if (data.profitZones && data.profitZones.length > 0) {
        data.profitZones.forEach((zone, i) => {
            traces.push({
                x: zone.time,
                y: zone.price,
                mode: 'markers',
                type: 'scatter',
                name: `Profit Zone ${i + 1}`,
                marker: {
                    size: 15,
                    color: '#00ff88',
                    symbol: 'star',
                    line: {
                        color: '#ffffff',
                        width: 1
                    }
                }
            });
        });
    }
    
    const layout = {
        title: {
            text: 'Future Cone: Market Trajectory & System Influence',
            font: { color: '#00d4ff', size: 16 }
        },
        xaxis: {
            title: 'Time (hours ahead)',
            color: '#a0a0a0',
            gridcolor: 'rgba(255,255,255,0.1)'
        },
        yaxis: {
            title: 'Price ($)',
            color: '#a0a0a0',
            gridcolor: 'rgba(255,255,255,0.1)'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0.3)',
        font: { color: '#e0e0e0' },
        legend: {
            font: { color: '#a0a0a0' },
            bgcolor: 'rgba(0,0,0,0.5)',
            bordercolor: 'rgba(255,255,255,0.2)',
            borderwidth: 1
        },
        margin: { l: 60, r: 20, t: 50, b: 50 },
        hovermode: 'closest'
    };
    
    const config = {
        displayModeBar: false,
        responsive: true
    };
    
    Plotly.newPlot(container, traces, layout, config);
    
    futureCone = { container, traces, layout };
}

function generateFutureCone() {
    // Generate future price trajectory with uncertainty
    const hoursAhead = 24;
    const currentPrice = 50000; // BTC price example
    const timeSteps = 100;
    
    const trajectory = { time: [], price: [] };
    const upper = { time: [], price: [] };
    const lower = { time: [], price: [] };
    const profitZones = [];
    
    // Generate trajectory (upward trend with volatility)
    for (let i = 0; i <= timeSteps; i++) {
        const t = (i / timeSteps) * hoursAhead;
        const progress = i / timeSteps;
        
        // Base trajectory (upward with some noise)
        const trend = currentPrice * (1 + progress * 0.02); // 2% increase over 24h
        const volatility = Math.sin(progress * Math.PI * 4) * 100; // Oscillations
        const noise = (Math.random() - 0.5) * 50;
        
        const price = trend + volatility + noise;
        
        // Uncertainty bounds (cone widens over time)
        const uncertainty = 500 * (1 + progress * 2); // Increasing uncertainty
        const upperPrice = price + uncertainty;
        const lowerPrice = price - uncertainty;
        
        trajectory.time.push(t);
        trajectory.price.push(price);
        upper.time.push(t);
        upper.price.push(upperPrice);
        lower.time.push(t);
        lower.price.push(lowerPrice);
        
        // Mark high-probability profit zones (where price spikes)
        if (i > 0 && price > trajectory.price[i - 1] + 200 && Math.random() > 0.9) {
            profitZones.push({
                time: [t],
                price: [price]
            });
        }
    }
    
    return { trajectory, upper, lower, profitZones };
}

// Update data from API
function startDataUpdates() {
    // Update metrics
    updateInterval = setInterval(() => {
        fetchMetrics();
        updateVisualizations();
    }, 5000); // Update every 5 seconds
    
    // Initial fetch
    fetchMetrics();
}

async function fetchMetrics() {
    try {
        // Fetch meta-learning stats
        const metaResponse = await fetch(`${API_BASE}/meta-learning/state`);
        if (metaResponse.ok) {
            const metaData = await metaResponse.json();
            updateMetric('meta-scenarios', Math.floor(metaData.scenarios_per_second || 0).toLocaleString());
            updateMetric('patches-count', metaData.patches_applied || 0);
        }
        
        // Fetch causal inference stats
        const causalResponse = await fetch(`${API_BASE}/causal-inference/stats`);
        if (causalResponse.ok) {
            const causalData = await causalResponse.json();
            updateMetric('causal-count', Object.keys(causalData.relationships || {}).length);
            updateMetric('pre-echo-count', causalData.pre_echo_signals?.length || 0);
        }
        
        // Fetch multimodal swarm stats
        const swarmResponse = await fetch(`${API_BASE}/multimodal-swarm/stats`);
        if (swarmResponse.ok) {
            const swarmData = await swarmResponse.json();
            updateMetric('tensor-size', swarmData.tensor_size || 18);
        }
        
    } catch (error) {
        console.error('Error fetching metrics:', error);
    }
}

function updateMetric(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function updateVisualizations() {
    // Update probability manifold with new data
    if (probabilityManifold) {
        const newData = generateProbabilityManifold();
        const update = {
            z: [newData.z]
        };
        Plotly.update(probabilityManifold.container, update, probabilityManifold.layout);
    }
    
    // Update future cone with new trajectory
    if (futureCone) {
        const newData = generateFutureCone();
        const update = {
            x: [[newData.trajectory.time], [newData.upper.time], [newData.lower.time]],
            y: [[newData.trajectory.price], [newData.upper.price], [newData.lower.price]]
        };
        Plotly.update(futureCone.container, update, futureCone.layout);
    }
}

function updateSystemStatus() {
    // Update status indicators (simulated - in production, check actual system status)
    const statuses = ['meta-status', 'causal-status', 'swarm-status', 'ui-status'];
    
    statuses.forEach(statusId => {
        const statusDot = document.getElementById(statusId);
        if (statusDot) {
            // Simulate status (in production, check actual system health)
            const isHealthy = Math.random() > 0.1; // 90% uptime
            statusDot.className = 'status-dot' + (isHealthy ? '' : ' warning');
        }
    });
    
    // Update periodically
    setTimeout(updateSystemStatus, 3000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});
