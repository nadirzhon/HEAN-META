/**
 * HEAN Trading Command Center - Main Application
 */

// Global State
const state = {
    currentPage: 'dashboard',
    engineStatus: null,
    settings: null,
    positions: [],
    orders: [],
    strategies: [],
    riskStatus: null,
    eventStream: null,
    logStream: null,
    events: [],
    logs: [],
    autoScrollEvents: true,
    autoScrollLogs: true,
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('HEAN Command Center: Initializing...');
    
    // Check if API client is available
    if (typeof api === 'undefined') {
        console.error('API client not found! Make sure api-client.js is loaded before command-center.js');
        showToast('API client not loaded. Please refresh the page.', 'error');
        return;
    }
    
    initializeApp().catch(error => {
        console.error('Failed to initialize app:', error);
        showToast('Failed to initialize application', 'error');
    });
});

async function initializeApp() {
    try {
        console.log('Setting up UI components...');
        setupNavigation();
        setupButtons();
        setupCommandPalette();
        setupThemeToggle();
        setupHotkeys();
        setupConfirmModal();
        
        console.log('Loading initial data...');
        // Load initial data
        await loadSettings();
        await loadEngineStatus();
        await updateIndicators();
        
        console.log('Starting polling and streams...');
        // Start polling
        startPolling();
        
        // Start SSE streams
        startEventStream();
        startLogStream();
        
        console.log('HEAN Command Center: Initialized successfully');
        showToast('Application loaded', 'success');
    } catch (error) {
        console.error('Error during initialization:', error);
        showToast(`Initialization error: ${error.message}`, 'error');
        throw error;
    }
}

// Navigation
function setupNavigation() {
    console.log('Setting up navigation...');
    const navItems = document.querySelectorAll('.nav-item');
    console.log(`Found ${navItems.length} nav items`);
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            console.log(`Navigating to page: ${page}`);
            switchPage(page);
        });
    });
}

function switchPage(page) {
    state.currentPage = page;
    
    // Update nav
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });
    
    // Update pages
    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === `page-${page}`);
    });
    
    // Load page data
    loadPageData(page);
}

function loadPageData(page) {
    switch(page) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'trading':
            loadTrading();
            break;
        case 'strategies':
            loadStrategies();
            break;
        case 'analytics':
            loadAnalytics();
            break;
        case 'risk':
            loadRisk();
            break;
        case 'logs':
            loadLogs();
            break;
        case 'settings':
            loadSettings();
            break;
    }
}

// Buttons
function setupButtons() {
    console.log('Setting up button handlers...');
    
    // Engine controls
    const startBtn = document.getElementById('btn-start-engine');
    const stopBtn = document.getElementById('btn-stop-engine');
    const pauseBtn = document.getElementById('btn-pause-engine');
    const resumeBtn = document.getElementById('btn-resume-engine');
    
    if (startBtn) {
        startBtn.addEventListener('click', () => handleStartEngine());
        console.log('Start engine button handler attached');
    } else {
        console.warn('btn-start-engine not found');
    }
    
    if (stopBtn) {
        stopBtn.addEventListener('click', () => handleStopEngine());
        console.log('Stop engine button handler attached');
    } else {
        console.warn('btn-stop-engine not found');
    }
    
    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => handlePauseEngine());
        console.log('Pause engine button handler attached');
    }
    
    if (resumeBtn) {
        resumeBtn.addEventListener('click', () => handleResumeEngine());
        console.log('Resume engine button handler attached');
    }
    
    // Actions
    const reconcileBtn = document.getElementById('btn-reconcile');
    const smokeTestBtn = document.getElementById('btn-smoke-test');
    const placeTestOrderBtn = document.getElementById('btn-place-test-order');
    const closePositionBtn = document.getElementById('btn-close-position');
    const cancelAllBtn = document.getElementById('btn-cancel-all');
    
    if (reconcileBtn) {
        reconcileBtn.addEventListener('click', () => handleReconcile());
    }
    if (smokeTestBtn) {
        smokeTestBtn.addEventListener('click', () => handleSmokeTest());
    }
    if (placeTestOrderBtn) {
        placeTestOrderBtn.addEventListener('click', () => handlePlaceTestOrder());
    }
    if (closePositionBtn) {
        closePositionBtn.addEventListener('click', () => handleClosePosition());
    }
    if (cancelAllBtn) {
        cancelAllBtn.addEventListener('click', () => handleCancelAll());
    }
    
    // Analytics
    const backtestBtn = document.getElementById('btn-run-backtest');
    const evaluateBtn = document.getElementById('btn-run-evaluate');
    
    if (backtestBtn) {
        backtestBtn.addEventListener('click', () => handleRunBacktest());
    }
    if (evaluateBtn) {
        evaluateBtn.addEventListener('click', () => handleRunEvaluate());
    }
    
    // UI controls
    const clearEventsBtn = document.getElementById('btn-clear-events');
    const clearLogsBtn = document.getElementById('btn-clear-logs');
    const autoScrollToggle = document.getElementById('toggle-auto-scroll');
    
    if (clearEventsBtn) {
        clearEventsBtn.addEventListener('click', () => clearEvents());
    }
    if (clearLogsBtn) {
        clearLogsBtn.addEventListener('click', () => clearLogs());
    }
    if (autoScrollToggle) {
        autoScrollToggle.addEventListener('change', (e) => {
            state.autoScrollEvents = e.target.checked;
        });
    }
    
    console.log('Button handlers setup complete');
}

// Engine Handlers
async function handleStartEngine() {
    console.log('handleStartEngine called');
    
    if (typeof api === 'undefined') {
        console.error('API client not available');
        showToast('API client not available. Please refresh the page.', 'error');
        return;
    }
    
    const needsConfirm = state.settings?.is_live;
    
    if (needsConfirm) {
        const confirmed = await showConfirmDanger({
            title: 'Start Engine (LIVE)',
            message: 'This will start the trading engine in LIVE mode. Real orders will be placed.',
            action: 'start_engine',
        });
        if (!confirmed) return;
    }
    
    try {
        showLoading('Starting engine...');
        console.log('Calling api.startEngine...');
        const result = await api.startEngine(needsConfirm ? 'I_UNDERSTAND_LIVE_TRADING' : null);
        console.log('Engine start result:', result);
        showToast('Engine started successfully', 'success');
        await loadEngineStatus();
    } catch (error) {
        console.error('Failed to start engine:', error);
        showToast(`Failed to start engine: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

async function handleStopEngine() {
    try {
        showLoading('Stopping engine...');
        await api.stopEngine();
        showToast('Engine stopped', 'success');
        await loadEngineStatus();
    } catch (error) {
        showToast(`Failed to stop engine: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

async function handlePauseEngine() {
    try {
        await api.pauseEngine();
        showToast('Engine paused', 'success');
        await loadEngineStatus();
    } catch (error) {
        showToast(`Failed to pause engine: ${error.message}`, 'error');
    }
}

async function handleResumeEngine() {
    try {
        await api.resumeEngine();
        showToast('Engine resumed', 'success');
        await loadEngineStatus();
    } catch (error) {
        showToast(`Failed to resume engine: ${error.message}`, 'error');
    }
}

async function handleReconcile() {
    try {
        showLoading('Reconciling...');
        await api.reconcileNow();
        showToast('Reconcile completed', 'success');
        await loadTrading();
    } catch (error) {
        showToast(`Reconcile failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

async function handleSmokeTest() {
    try {
        showLoading('Running smoke test...');
        const result = await api.runSmokeTest();
        showToast('Smoke test completed', result.success ? 'success' : 'warning');
    } catch (error) {
        showToast(`Smoke test failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// Data Loading
async function loadSettings() {
    try {
        state.settings = await api.getSettings();
        updateSettingsDisplay();
    } catch (error) {
        console.error('Failed to load settings:', error);
    }
}

async function loadEngineStatus() {
    try {
        state.engineStatus = await api.getEngineStatus();
        updateEngineStatus();
    } catch (error) {
        console.error('Failed to load engine status:', error);
    }
}

function updateEngineStatus() {
    const status = state.engineStatus;
    if (!status) return;
    
    // Update buttons
    const startBtn = document.getElementById('btn-start-engine');
    const stopBtn = document.getElementById('btn-stop-engine');
    const pauseBtn = document.getElementById('btn-pause-engine');
    const resumeBtn = document.getElementById('btn-resume-engine');
    
    if (status.running) {
        startBtn?.setAttribute('disabled', '');
        stopBtn?.removeAttribute('disabled');
        pauseBtn?.removeAttribute('disabled');
        resumeBtn?.setAttribute('disabled', '');
    } else {
        startBtn?.removeAttribute('disabled');
        stopBtn?.setAttribute('disabled', '');
        pauseBtn?.setAttribute('disabled', '');
        resumeBtn?.setAttribute('disabled', '');
    }
    
    // Update indicators
    updateIndicators();
}

async function loadDashboard() {
    await loadEngineStatus();
    await loadMetrics();
}

async function loadMetrics() {
    const status = state.engineStatus;
    if (!status) return;
    
    document.getElementById('metric-equity')?.textContent = formatCurrency(status.equity || 0);
    document.getElementById('metric-daily-pnl')?.textContent = formatCurrency(status.daily_pnl || 0);
    document.getElementById('metric-initial-capital')?.textContent = formatCurrency(status.initial_capital || 0);
}

async function loadTrading() {
    try {
        state.positions = await api.getPositions();
        state.orders = await api.getOrders();
        updatePositionsTable();
        updateOrdersTable();
    } catch (error) {
        console.error('Failed to load trading data:', error);
    }
}

function updatePositionsTable() {
    const tbody = document.getElementById('positions-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (state.positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center">No open positions</td></tr>';
        return;
    }
    
    state.positions.forEach(pos => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${pos.symbol}</td>
            <td>${pos.side}</td>
            <td>${pos.size}</td>
            <td>${formatCurrency(pos.entry_price)}</td>
            <td>--</td>
            <td>${formatCurrency(pos.unrealized_pnl || 0)}</td>
            <td><button class="btn btn-sm btn-danger" onclick="handleClosePositionById('${pos.position_id}')">Close</button></td>
        `;
        tbody.appendChild(row);
    });
}

function updateOrdersTable() {
    const tbody = document.getElementById('orders-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (state.orders.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center">No orders</td></tr>';
        return;
    }
    
    state.orders.forEach(order => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${order.order_id.substring(0, 8)}...</td>
            <td>${order.symbol}</td>
            <td>${order.side}</td>
            <td>${order.size}</td>
            <td>${order.price || 'Market'}</td>
            <td>${order.status}</td>
            <td>--</td>
        `;
        tbody.appendChild(row);
    });
}

async function loadStrategies() {
    try {
        state.strategies = await api.getStrategies();
        updateStrategiesList();
    } catch (error) {
        console.error('Failed to load strategies:', error);
    }
}

function updateStrategiesList() {
    const container = document.getElementById('strategies-list');
    if (!container) return;
    
    if (state.strategies.length === 0) {
        container.innerHTML = '<div class="text-center">No strategies found</div>';
        return;
    }
    
    container.innerHTML = state.strategies.map(strategy => `
        <div class="section">
            <div class="section-header">
                <h3>${strategy.strategy_id}</h3>
                <label class="toggle">
                    <input type="checkbox" ${strategy.enabled ? 'checked' : ''} 
                           onchange="handleToggleStrategy('${strategy.strategy_id}', this.checked)">
                    <span>Enabled</span>
                </label>
            </div>
            <p>Type: ${strategy.type}</p>
        </div>
    `).join('');
}

async function loadAnalytics() {
    try {
        const summary = await api.getAnalyticsSummary();
        const blocks = await api.getBlockedSignals();
        updateAnalyticsSummary(summary);
        updateBlockedSignals(blocks);
    } catch (error) {
        console.error('Failed to load analytics:', error);
    }
}

function updateAnalyticsSummary(summary) {
    const container = document.getElementById('analytics-summary');
    if (!container) return;
    
    container.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">${summary.total_trades || 0}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">${(summary.win_rate || 0).toFixed(2)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">${(summary.profit_factor || 0).toFixed(2)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">${(summary.max_drawdown_pct || 0).toFixed(2)}%</div>
            </div>
        </div>
    `;
}

function updateBlockedSignals(blocks) {
    const container = document.getElementById('analytics-blocks');
    if (!container) return;
    
    container.innerHTML = `
        <div class="metric-card">
            <div class="metric-label">Total Blocks</div>
            <div class="metric-value">${blocks.total_blocks || 0}</div>
        </div>
        <div class="mt-2">
            <h3>Top Reasons</h3>
            <ul>
                ${(blocks.top_reasons || []).slice(0, 5).map(r => `<li>${r.code}: ${r.count}</li>`).join('')}
            </ul>
        </div>
    `;
}

async function loadRisk() {
    try {
        state.riskStatus = await api.getRiskStatus();
        const limits = await api.getRiskLimits();
        updateRiskStatus(state.riskStatus);
        updateRiskLimits(limits);
    } catch (error) {
        console.error('Failed to load risk data:', error);
    }
}

function updateRiskStatus(status) {
    const container = document.getElementById('risk-status');
    if (!container) return;
    
    container.innerHTML = `
        <div class="metric-card">
            <div class="metric-label">Killswitch</div>
            <div class="metric-value">${status.killswitch_triggered ? 'TRIGGERED' : 'SAFE'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Stop Trading</div>
            <div class="metric-value">${status.stop_trading ? 'YES' : 'NO'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Equity</div>
            <div class="metric-value">${formatCurrency(status.equity || 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Drawdown</div>
            <div class="metric-value">${(status.drawdown_pct || 0).toFixed(2)}%</div>
        </div>
    `;
}

function updateRiskLimits(limits) {
    const container = document.getElementById('risk-limits');
    if (!container) return;
    
    container.innerHTML = `
        <div class="metric-card">
            <div class="metric-label">Max Open Positions</div>
            <div class="metric-value">${limits.max_open_positions || 0}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Daily Attempts</div>
            <div class="metric-value">${limits.max_daily_attempts || 0}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Exposure</div>
            <div class="metric-value">${formatCurrency(limits.max_exposure_usd || 0)}</div>
        </div>
    `;
}

function loadLogs() {
    // Logs are loaded via SSE stream
    updateLogFeed();
}

function updateSettingsDisplay() {
    const container = document.getElementById('settings-display');
    if (!container || !state.settings) return;
    
    container.innerHTML = `
        <div class="metric-card">
            <div class="metric-label">Trading Mode</div>
            <div class="metric-value">${state.settings.trading_mode || 'N/A'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Is Live</div>
            <div class="metric-value">${state.settings.is_live ? 'YES' : 'NO'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Dry Run</div>
            <div class="metric-value">${state.settings.dry_run ? 'YES' : 'NO'}</div>
        </div>
    `;
}

// SSE Streams
function startEventStream() {
    if (state.eventStream) {
        state.eventStream.close();
    }
    
    state.eventStream = api.createEventStream((event) => {
        state.events.push(event);
        if (state.events.length > 1000) {
            state.events.shift();
        }
        addEventToFeed(event);
    });
}

function startLogStream() {
    if (state.logStream) {
        state.logStream.close();
    }
    
    state.logStream = api.createLogStream((log) => {
        state.logs.push(log);
        if (state.logs.length > 2000) {
            state.logs.shift();
        }
        addLogToFeed(log);
    });
}

function addEventToFeed(event) {
    const feed = document.getElementById('event-feed');
    if (!feed) return;
    
    const item = document.createElement('div');
    item.className = 'event-item';
    item.innerHTML = `
        <span class="event-time">${new Date(event.timestamp).toLocaleTimeString()}</span>
        <span class="event-type">${event.event}</span>
        <span>${JSON.stringify(event.data)}</span>
    `;
    
    feed.appendChild(item);
    
    if (state.autoScrollEvents) {
        feed.scrollTop = feed.scrollHeight;
    }
}

function addLogToFeed(log) {
    const feed = document.getElementById('log-feed');
    if (!feed) return;
    
    const item = document.createElement('div');
    item.className = `log-item log-level-${log.level}`;
    item.textContent = `[${new Date(log.timestamp).toLocaleTimeString()}] [${log.level.toUpperCase()}] ${log.message}`;
    
    feed.appendChild(item);
    
    if (state.autoScrollLogs) {
        feed.scrollTop = feed.scrollHeight;
    }
}

function clearEvents() {
    state.events = [];
    document.getElementById('event-feed').innerHTML = '';
}

function clearLogs() {
    state.logs = [];
    document.getElementById('log-feed').innerHTML = '';
}

function updateLogFeed() {
    const feed = document.getElementById('log-feed');
    if (!feed) return;
    
    feed.innerHTML = '';
    state.logs.forEach(log => addLogToFeed(log));
}

// Polling
function startPolling() {
    setInterval(async () => {
        await loadEngineStatus();
        await updateIndicators();
        
        if (state.currentPage === 'dashboard') {
            await loadMetrics();
        } else if (state.currentPage === 'trading') {
            await loadTrading();
        }
    }, 5000); // Poll every 5 seconds
}

function updateIndicators() {
    const status = state.engineStatus;
    const settings = state.settings;
    
    // Mode indicator
    const modeIndicator = document.getElementById('indicator-mode');
    if (modeIndicator && settings) {
        modeIndicator.textContent = settings.is_live ? 'LIVE' : 'PAPER';
        modeIndicator.className = `indicator ${settings.is_live ? 'indicator-danger' : 'indicator-safe'}`;
    }
    
    // Engine indicator
    const engineIndicator = document.getElementById('indicator-engine');
    if (engineIndicator && status) {
        engineIndicator.textContent = status.running ? 'ONLINE' : 'OFFLINE';
        engineIndicator.className = `indicator ${status.running ? 'indicator-online' : 'indicator-offline'}`;
    }
    
    // Latency indicator
    const latencyIndicator = document.getElementById('indicator-latency');
    if (latencyIndicator && api.latency) {
        latencyIndicator.textContent = `${Math.round(api.latency)}ms`;
    }
    
    // Risk indicator
    const riskIndicator = document.getElementById('indicator-risk');
    if (riskIndicator && state.riskStatus) {
        const risk = state.riskStatus;
        if (risk.killswitch_triggered) {
            riskIndicator.textContent = 'KILL';
            riskIndicator.className = 'indicator indicator-danger';
        } else if (risk.stop_trading) {
            riskIndicator.textContent = 'PAUSED';
            riskIndicator.className = 'indicator indicator-warning';
        } else {
            riskIndicator.textContent = 'SAFE';
            riskIndicator.className = 'indicator indicator-safe';
        }
    }
}

// Command Palette
function setupCommandPalette() {
    console.log('Setting up command palette...');
    const palette = document.getElementById('command-palette');
    const toggle = document.getElementById('btn-command-palette');
    const input = document.getElementById('command-input');
    
    if (!palette || !toggle) {
        console.warn('Command palette elements not found');
        return;
    }
    
    toggle.addEventListener('click', () => {
        palette.classList.add('active');
        input?.focus();
    });
    
    const closeBtn = palette.querySelector('.modal-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            palette.classList.remove('active');
        });
    }
    
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'k') {
            e.preventDefault();
            palette.classList.toggle('active');
            if (palette.classList.contains('active')) {
                input?.focus();
            }
        }
    });
    
    console.log('Command palette setup complete');
}

// Theme Toggle
function setupThemeToggle() {
    console.log('Setting up theme toggle...');
    const toggle = document.getElementById('btn-theme-toggle');
    if (!toggle) {
        console.warn('Theme toggle button not found');
        return;
    }
    
    toggle.addEventListener('click', () => {
        document.body.classList.toggle('theme-dark');
        document.body.classList.toggle('theme-light');
        const isDark = document.body.classList.contains('theme-dark');
        console.log(`Theme switched to: ${isDark ? 'dark' : 'light'}`);
    });
    
    console.log('Theme toggle setup complete');
}

// Hotkeys
function setupHotkeys() {
    document.addEventListener('keydown', (e) => {
        // Ctrl+K for command palette (handled above)
        // Add more hotkeys as needed
    });
}

// Confirm Modal
function setupConfirmModal() {
    console.log('Setting up confirm modal...');
    const modal = document.getElementById('confirm-danger-modal');
    const cancelBtn = document.getElementById('confirm-cancel');
    const submitBtn = document.getElementById('confirm-submit');
    const checkbox = document.getElementById('confirm-checkbox');
    const phraseInput = document.getElementById('confirm-phrase-input');
    
    if (!modal || !cancelBtn || !submitBtn) {
        console.warn('Confirm modal elements not found');
        return;
    }
    
    let resolveCallback = null;
    
    cancelBtn.addEventListener('click', () => {
        modal.classList.remove('active');
        if (resolveCallback) resolveCallback(false);
    });
    
    submitBtn.addEventListener('click', () => {
        if (checkbox?.checked && phraseInput?.value === 'I_UNDERSTAND_LIVE_TRADING') {
            modal.classList.remove('active');
            if (resolveCallback) resolveCallback(true);
        }
    });
    
    if (checkbox) {
        checkbox.addEventListener('change', () => {
            updateConfirmButton();
        });
    }
    
    if (phraseInput) {
        phraseInput.addEventListener('input', () => {
            updateConfirmButton();
        });
    }
    
    function updateConfirmButton() {
        const enabled = checkbox?.checked && phraseInput?.value === 'I_UNDERSTAND_LIVE_TRADING';
        submitBtn.toggleAttribute('disabled', !enabled);
    }
    
    console.log('Confirm modal setup complete');
}

async function showConfirmDanger({ title, message, action }) {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirm-danger-modal');
        const titleEl = document.getElementById('confirm-title');
        const messageEl = document.getElementById('confirm-message');
        const dryRunEl = document.getElementById('confirm-dry-run');
        const liveConfirmEl = document.getElementById('confirm-live-confirm');
        const modeEl = document.getElementById('confirm-mode');
        const checkbox = document.getElementById('confirm-checkbox');
        const phraseInput = document.getElementById('confirm-phrase-input');
        
        titleEl.textContent = title;
        messageEl.textContent = message;
        dryRunEl.textContent = state.settings?.dry_run ? 'true' : 'false';
        liveConfirmEl.textContent = state.settings?.live_confirm ? 'true' : 'false';
        modeEl.textContent = state.settings?.trading_mode || 'N/A';
        
        checkbox.checked = false;
        phraseInput.value = '';
        
        modal.classList.add('active');
        
        const submitBtn = document.getElementById('confirm-submit');
        const cancelBtn = document.getElementById('confirm-cancel');
        
        const handleSubmit = () => {
            if (checkbox.checked && phraseInput.value === 'I_UNDERSTAND_LIVE_TRADING') {
                modal.classList.remove('active');
                submitBtn.removeEventListener('click', handleSubmit);
                cancelBtn.removeEventListener('click', handleCancel);
                resolve(true);
            }
        };
        
        const handleCancel = () => {
            modal.classList.remove('active');
            submitBtn.removeEventListener('click', handleSubmit);
            cancelBtn.removeEventListener('click', handleCancel);
            resolve(false);
        };
        
        submitBtn.addEventListener('click', handleSubmit);
        cancelBtn.addEventListener('click', handleCancel);
    });
}

// UI Helpers
function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    const messageEl = overlay?.querySelector('.loading-message');
    if (messageEl) messageEl.textContent = message;
    overlay?.classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-overlay')?.classList.add('hidden');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) {
        console.warn('Toast container not found, creating fallback');
        // Fallback: use console and alert for critical errors
        console[type === 'error' ? 'error' : 'log'](`[${type.toUpperCase()}] ${message}`);
        if (type === 'error') {
            alert(`Error: ${message}`);
        }
        return;
    }
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Add click to dismiss
    toast.addEventListener('click', () => toast.remove());
    
    container.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.remove();
        }
    }, 5000);
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
    }).format(value);
}

// Global handlers (for inline onclick)
window.handleClosePositionById = async function(positionId) {
    const needsConfirm = state.settings?.is_live;
    if (needsConfirm) {
        const confirmed = await showConfirmDanger({
            title: 'Close Position (LIVE)',
            message: `Close position ${positionId}?`,
            action: 'close_position',
        });
        if (!confirmed) return;
    }
    
    try {
        await api.closePosition(positionId, needsConfirm ? 'I_UNDERSTAND_LIVE_TRADING' : null);
        showToast('Position closed', 'success');
        await loadTrading();
    } catch (error) {
        showToast(`Failed to close position: ${error.message}`, 'error');
    }
};

window.handleToggleStrategy = async function(strategyId, enabled) {
    try {
        await api.enableStrategy(strategyId, enabled);
        showToast(`Strategy ${enabled ? 'enabled' : 'disabled'}`, 'success');
    } catch (error) {
        showToast(`Failed to toggle strategy: ${error.message}`, 'error');
    }
};

// Placeholder handlers
async function handlePlaceTestOrder() {
    showToast('Test order placement not yet implemented', 'info');
}

async function handleClosePosition() {
    showToast('Please select a position from the table', 'info');
}

async function handleCancelAll() {
    const needsConfirm = state.settings?.is_live;
    if (needsConfirm) {
        const confirmed = await showConfirmDanger({
            title: 'Cancel All Orders (LIVE)',
            message: 'Cancel all open orders?',
            action: 'cancel_all',
        });
        if (!confirmed) return;
    }
    
    try {
        await api.cancelAllOrders(needsConfirm ? 'I_UNDERSTAND_LIVE_TRADING' : null);
        showToast('All orders cancelled', 'success');
        await loadTrading();
    } catch (error) {
        showToast(`Failed to cancel orders: ${error.message}`, 'error');
    }
}

async function handleRunBacktest() {
    showToast('Backtest not yet implemented', 'info');
}

async function handleRunEvaluate() {
    showToast('Evaluate not yet implemented', 'info');
}

