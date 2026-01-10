// HEAN Dashboard JavaScript
// Use relative path for API (proxied through nginx)
const API_BASE = '/api';

// State
let currentPage = 'dashboard';
let pollingInterval = null;
let events = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('HEAN Dashboard: Initializing...');
    try {
        initializeNavigation();
        initializeButtons();
        loadDashboard();
        startPolling();
        console.log('HEAN Dashboard: Initialized successfully');
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
        alert(`Ошибка инициализации: ${error.message}`);
    }
});

// Navigation
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const page = btn.dataset.page;
            switchPage(page);
        });
    });
}

function switchPage(page) {
    // Update nav
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.page === page);
    });

    // Update pages
    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === `page-${page}`);
    });

    currentPage = page;

    // Load page-specific data
    if (page === 'dashboard') {
        loadDashboard();
    } else if (page === 'orders') {
        loadOrders();
    } else if (page === 'positions') {
        loadPositions();
    } else if (page === 'depth') {
        loadDepth();
    } else if (page === 'settings') {
        loadSettings();
    }
}

// Buttons
function initializeButtons() {
    console.log('Setting up button handlers...');
    
    // Engine controls
    const startBtn = document.getElementById('btn-start-engine');
    const stopBtn = document.getElementById('btn-stop-engine');
    const refreshBtn = document.getElementById('btn-refresh');
    
    if (startBtn) {
        startBtn.addEventListener('click', startEngine);
        console.log('Start engine button handler attached');
    } else {
        console.warn('btn-start-engine not found');
    }
    
    if (stopBtn) {
        stopBtn.addEventListener('click', stopEngine);
        console.log('Stop engine button handler attached');
    } else {
        console.warn('btn-stop-engine not found');
    }
    
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadDashboard);
        console.log('Refresh button handler attached');
    }

    // Smoke test
    const smokeTestBtn = document.getElementById('btn-run-smoke-test');
    if (smokeTestBtn) {
        smokeTestBtn.addEventListener('click', runSmokeTest);
    }

    // Test order
    const testOrderBtn = document.getElementById('btn-place-test-order');
    if (testOrderBtn) {
        testOrderBtn.addEventListener('click', placeTestOrder);
    }

    // Refresh positions
    const refreshPositionsBtn = document.getElementById('btn-refresh-positions');
    if (refreshPositionsBtn) {
        refreshPositionsBtn.addEventListener('click', loadPositions);
    }

    // Depth visualization
    const refreshDepthBtn = document.getElementById('btn-refresh-depth');
    if (refreshDepthBtn) {
        refreshDepthBtn.addEventListener('click', loadDepth);
    }

    const depthSymbolSelect = document.getElementById('depth-symbol-select');
    if (depthSymbolSelect) {
        depthSymbolSelect.addEventListener('change', loadDepth);
    }

    // Order filter
    const orderFilter = document.getElementById('order-filter');
    if (orderFilter) {
        orderFilter.addEventListener('change', (e) => {
            loadOrders(e.target.value);
        });
    }

    // Clear logs
    const clearLogsBtn = document.getElementById('btn-clear-logs');
    if (clearLogsBtn) {
        clearLogsBtn.addEventListener('click', () => {
            events = [];
            updateLogs();
        });
    }
    
    console.log('Button handlers setup complete');
}

// API Calls
async function apiCall(endpoint, options = {}) {
    try {
        const url = `${API_BASE}${endpoint}`;
        console.log(`API ${options.method || 'GET'} ${url}`, options.body ? JSON.parse(options.body) : '');
        
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });

        if (!response.ok) {
            let errorData;
            try {
                errorData = await response.json();
            } catch {
                errorData = { detail: response.statusText || `HTTP ${response.status}` };
            }
            const errorMessage = errorData.detail || errorData.message || `HTTP ${response.status}`;
            console.error(`API error ${response.status}: ${errorMessage}`);
            throw new Error(errorMessage);
        }

        const result = await response.json();
        console.log(`API ${options.method || 'GET'} ${url} success:`, result);
        return result;
    } catch (error) {
        console.error('API call failed:', error);
        if (error instanceof TypeError && error.message.includes('fetch')) {
            const networkError = 'Network error: Unable to connect to API server. Make sure the backend is running.';
            addEvent('error', networkError);
            throw new Error(networkError);
        }
        addEvent('error', `API Error: ${error.message}`);
        throw error;
    }
}

// Engine Controls
async function startEngine() {
    console.log('startEngine called');
    try {
        const result = await apiCall('/engine/start', {
            method: 'POST',
            body: JSON.stringify({ confirm_phrase: null }),
        });

        console.log('Engine start result:', result);
        addEvent('info', `Engine started: ${result.message || 'Success'}`);
        updateEngineStatus(true);
        loadDashboard();
    } catch (error) {
        console.error('Failed to start engine:', error);
        addEvent('error', `Failed to start engine: ${error.message}`);
    }
}

async function stopEngine() {
    try {
        const result = await apiCall('/engine/stop', {
            method: 'POST',
        });

        addEvent('info', `Engine stopped: ${result.message}`);
        updateEngineStatus(false);
        loadDashboard();
    } catch (error) {
        addEvent('error', `Failed to stop engine: ${error.message}`);
    }
}

// Dashboard
async function loadDashboard() {
    try {
        // Get engine status
        const status = await apiCall('/engine/status');
        updateEngineStatus(status.running);
        updateMetrics(status);

        // Get positions count
        const positionsData = await apiCall('/positions');
        document.getElementById('metric-positions-count').textContent = positionsData.positions?.length || 0;

        // Get health
        const health = await apiCall('/health');
        updateHealthChecks(health);
    } catch (error) {
        console.error('Failed to load dashboard:', error);
    }
}

function updateEngineStatus(running) {
    const statusEl = document.getElementById('engine-status');
    const startBtn = document.getElementById('btn-start-engine');
    const stopBtn = document.getElementById('btn-stop-engine');

    if (running) {
        statusEl.textContent = 'Running';
        statusEl.className = 'status-badge status-running';
        startBtn.disabled = true;
        stopBtn.disabled = false;
    } else {
        statusEl.textContent = 'Stopped';
        statusEl.className = 'status-badge status-stopped';
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

function updateMetrics(status) {
    document.getElementById('metric-equity').textContent = `$${status.equity?.toFixed(2) || '0.00'}`;
    document.getElementById('metric-daily-pnl').textContent = `$${status.daily_pnl?.toFixed(2) || '0.00'}`;
    document.getElementById('metric-initial-capital').textContent = `$${status.initial_capital?.toFixed(2) || '0.00'}`;

    // Update trading mode
    const modeEl = document.getElementById('trading-mode');
    if (status.trading_mode === 'live') {
        modeEl.textContent = 'Live';
        modeEl.className = 'mode-badge mode-live';
    } else {
        modeEl.textContent = 'Paper';
        modeEl.className = 'mode-badge mode-paper';
    }
}

function updateHealthChecks(health) {
    const container = document.getElementById('health-checks');
    container.innerHTML = `
        <div class="health-item">
            <span>System Status</span>
            <span class="health-status ${health.status === 'healthy' ? 'health-ok' : 'health-error'}">
                ${health.status}
            </span>
        </div>
        <div class="health-item">
            <span>Trading Mode</span>
            <span>${health.trading_mode}</span>
        </div>
        <div class="health-item">
            <span>Live Trading</span>
            <span>${health.is_live ? 'Yes' : 'No'}</span>
        </div>
    `;
}

// Orders
async function loadOrders(status = 'all') {
    try {
        const data = await apiCall(`/orders?status=${status}`);
        const tbody = document.getElementById('orders-tbody');
        
        if (!data.orders || data.orders.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading">No orders</td></tr>';
            return;
        }

        tbody.innerHTML = data.orders.map(order => `
            <tr>
                <td>${order.order_id.substring(0, 8)}...</td>
                <td>${order.symbol}</td>
                <td>${order.side}</td>
                <td>${order.size}</td>
                <td>$${order.price?.toFixed(2) || 'N/A'}</td>
                <td>${order.status}</td>
                <td>${new Date(order.timestamp).toLocaleString()}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Failed to load orders:', error);
    }
}

// Positions
async function loadPositions() {
    try {
        const data = await apiCall('/positions');
        const tbody = document.getElementById('positions-tbody');
        
        if (!data.positions || data.positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="loading">No positions</td></tr>';
            return;
        }

        tbody.innerHTML = data.positions.map(pos => `
            <tr>
                <td>${pos.symbol}</td>
                <td>${pos.side || 'N/A'}</td>
                <td>${pos.size}</td>
                <td>$${pos.entry_price?.toFixed(2) || 'N/A'}</td>
                <td class="${pos.unrealized_pnl >= 0 ? '' : 'text-red'}">$${pos.unrealized_pnl?.toFixed(2) || '0.00'}</td>
                <td>$${pos.realized_pnl?.toFixed(2) || '0.00'}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Failed to load positions:', error);
    }
}

// Depth Visualization
async function loadDepth() {
    try {
        const symbolSelect = document.getElementById('depth-symbol-select');
        const selectedSymbol = symbolSelect?.value || null;
        const endpoint = selectedSymbol ? `/orderbook-presence?symbol=${selectedSymbol}` : '/orderbook-presence';
        
        const data = await apiCall(endpoint);
        const visualization = document.getElementById('depth-visualization');
        const info = document.getElementById('depth-info');
        
        if (!visualization || !info) {
            return;
        }
        
        // Handle single symbol or list of symbols
        const presenceData = Array.isArray(data) ? data : (data && Object.keys(data).length > 0 ? [data] : []);
        
        if (presenceData.length === 0) {
            visualization.innerHTML = '<div class="loading">No active orders in orderbook</div>';
            info.innerHTML = '<div class="depth-info-item">No presence data available</div>';
            return;
        }
        
        // Build visualization HTML
        let visualizationHTML = '<div class="depth-chart">';
        
        for (const presence of presenceData) {
            const symbol = presence.symbol || 'UNKNOWN';
            const midPrice = presence.mid_price || 0;
            const bestBid = presence.best_bid || midPrice * 0.9999;
            const bestAsk = presence.best_ask || midPrice * 1.0001;
            const bidOrders = presence.bid_orders || [];
            const askOrders = presence.ask_orders || [];
            const numOrders = presence.num_orders || 0;
            const totalSize = presence.total_size || 0;
            
            visualizationHTML += `
                <div class="depth-symbol-section">
                    <h3>${symbol}</h3>
                    <div class="depth-orderbook">
                        <div class="depth-asks">
                            <div class="depth-header">Asks (Sell Orders)</div>
                            ${askOrders.length > 0 ? askOrders.map(order => `
                                <div class="depth-order-item depth-order-ask" 
                                     style="left: ${Math.min(100, (order.distance_from_mid_bps || 0) * 10)}%; 
                                            opacity: ${Math.min(1.0, (order.size || 0) / 10)};">
                                    <div class="depth-order-price">$${order.price?.toFixed(2) || 'N/A'}</div>
                                    <div class="depth-order-size">${order.size?.toFixed(4) || '0'}</div>
                                    <div class="depth-order-distance">${(order.distance_from_mid_bps || 0).toFixed(2)} bps</div>
                                </div>
                            `).join('') : '<div class="no-orders">No ask orders</div>'}
                            <div class="depth-best-price">Best Ask: $${bestAsk.toFixed(2)}</div>
                        </div>
                        
                        <div class="depth-mid">
                            <div class="depth-mid-price">Mid: $${midPrice.toFixed(2)}</div>
                            <div class="depth-spread">Spread: ${((bestAsk - bestBid) / midPrice * 10000).toFixed(2)} bps</div>
                        </div>
                        
                        <div class="depth-bids">
                            <div class="depth-header">Bids (Buy Orders)</div>
                            ${bidOrders.length > 0 ? bidOrders.map(order => `
                                <div class="depth-order-item depth-order-bid" 
                                     style="left: ${Math.min(100, Math.abs(order.distance_from_mid_bps || 0) * 10)}%; 
                                            opacity: ${Math.min(1.0, (order.size || 0) / 10)};">
                                    <div class="depth-order-price">$${order.price?.toFixed(2) || 'N/A'}</div>
                                    <div class="depth-order-size">${order.size?.toFixed(4) || '0'}</div>
                                    <div class="depth-order-distance">${Math.abs(order.distance_from_mid_bps || 0).toFixed(2)} bps</div>
                                </div>
                            `).join('') : '<div class="no-orders">No bid orders</div>'}
                            <div class="depth-best-price">Best Bid: $${bestBid.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        visualizationHTML += '</div>';
        visualization.innerHTML = visualizationHTML;
        
        // Build info HTML
        let infoHTML = '<div class="depth-info-header">Orderbook Presence Summary</div>';
        for (const presence of presenceData) {
            infoHTML += `
                <div class="depth-info-item">
                    <div class="depth-info-symbol">${presence.symbol || 'UNKNOWN'}</div>
                    <div class="depth-info-stats">
                        <div>Orders: ${presence.num_orders || 0}</div>
                        <div>Total Size: ${presence.total_size?.toFixed(4) || '0'}</div>
                        <div>Bid Orders: ${presence.bid_orders?.length || 0}</div>
                        <div>Ask Orders: ${presence.ask_orders?.length || 0}</div>
                    </div>
                </div>
            `;
        }
        info.innerHTML = infoHTML;
        
        // Update symbol select if needed
        if (symbolSelect && !selectedSymbol) {
            const symbols = presenceData.map(p => p.symbol).filter(Boolean);
            symbols.forEach(symbol => {
                if (!Array.from(symbolSelect.options).some(opt => opt.value === symbol)) {
                    const option = document.createElement('option');
                    option.value = symbol;
                    option.textContent = symbol;
                    symbolSelect.appendChild(option);
                }
            });
        }
        
    } catch (error) {
        console.error('Failed to load depth visualization:', error);
        const visualization = document.getElementById('depth-visualization');
        if (visualization) {
            visualization.innerHTML = `<div class="error">Failed to load depth data: ${error.message}</div>`;
        }
    }
}

// Settings
async function loadSettings() {
    try {
        const settings = await apiCall('/settings');
        const container = document.getElementById('settings-content');
        
        container.innerHTML = Object.entries(settings).map(([key, value]) => `
            <div class="setting-item">
                <div class="setting-label">${key}</div>
                <div class="setting-value">${JSON.stringify(value)}</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to load settings:', error);
    }
}

// Smoke Test
async function runSmokeTest() {
    try {
        addEvent('info', 'Running smoke test...');
        const result = await apiCall('/smoke-test/run', { method: 'POST' });
        
        if (result.success) {
            addEvent('info', `Smoke test passed: Order ${result.order_id}`);
        } else {
            addEvent('error', `Smoke test failed: ${result.error}`);
        }
    } catch (error) {
        addEvent('error', `Smoke test error: ${error.message}`);
    }
}

// Test Order
async function placeTestOrder() {
    try {
        const result = await apiCall('/orders/test', {
            method: 'POST',
            body: JSON.stringify({
                symbol: 'BTCUSDT',
                side: 'BUY',
                notional_usd: 5.0,
            }),
        });
        
        addEvent('info', `Test order placed: ${result.message}`);
        loadOrders();
    } catch (error) {
        addEvent('error', `Failed to place test order: ${error.message}`);
    }
}

// Events
function addEvent(level, message) {
    events.unshift({
        time: new Date().toISOString(),
        level,
        message,
    });
    
    // Keep only last 100 events
    if (events.length > 100) {
        events = events.slice(0, 100);
    }
    
    updateEvents();
    updateLogs();
}

function updateEvents() {
    const container = document.getElementById('events-list');
    if (!container) return;
    
    container.innerHTML = events.slice(0, 10).map(event => `
        <div class="event-item">
            <div class="event-time">${new Date(event.time).toLocaleString()}</div>
            <div class="event-message">[${event.level.toUpperCase()}] ${event.message}</div>
        </div>
    `).join('');
}

function updateLogs() {
    const container = document.getElementById('logs-container');
    if (!container) return;
    
    container.innerHTML = events.map(event => `
        <div class="log-entry">
            <span class="log-time">${new Date(event.time).toLocaleString()}</span>
            <span class="log-level-${event.level}">[${event.level.toUpperCase()}]</span>
            <span>${event.message}</span>
        </div>
    `).join('');
    
    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

// Polling
function startPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    pollingInterval = setInterval(() => {
        if (currentPage === 'dashboard') {
            loadDashboard();
        } else if (currentPage === 'orders') {
            loadOrders(document.getElementById('order-filter')?.value || 'all');
        } else if (currentPage === 'positions') {
            loadPositions();
        } else if (currentPage === 'depth') {
            loadDepth();
        }
    }, 5000); // Poll every 5 seconds
}

// Initial load
addEvent('info', 'Dashboard initialized');

