/**
 * Typed API Client for HEAN Trading System
 */

class HEANApiClient {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
        this.requestId = 0;
        this.latency = null;
    }

    async request(method, path, data = null) {
        const startTime = performance.now();
        const url = `${this.baseURL}${path}`;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            console.log(`API ${method} ${url}`, data ? { data } : '');
            const response = await fetch(url, options);
            const latency = performance.now() - startTime;
            this.latency = latency;

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
            console.log(`API ${method} ${url} success:`, result);
            return result;
        } catch (error) {
            console.error(`API request failed: ${method} ${path}`, error);
            // Re-throw with more context
            if (error instanceof TypeError && error.message.includes('fetch')) {
                throw new Error(`Network error: Unable to connect to API server. Make sure the backend is running.`);
            }
            throw error;
        }
    }

    // Health & System
    async getHealth() {
        return this.request('GET', '/health');
    }

    async getSettings() {
        return this.request('GET', '/settings');
    }

    // Engine
    async getEngineStatus() {
        return this.request('GET', '/engine/status');
    }

    async startEngine(confirmPhrase = null) {
        return this.request('POST', '/engine/start', { confirm_phrase: confirmPhrase });
    }

    async stopEngine() {
        return this.request('POST', '/engine/stop', {});
    }

    async pauseEngine() {
        return this.request('POST', '/engine/pause', {});
    }

    async resumeEngine() {
        return this.request('POST', '/engine/resume');
    }

    // Trading
    async getPositions() {
        return this.request('GET', '/orders/positions');
    }

    async getOrders(status = 'all') {
        return this.request('GET', `/orders?status=${status}`);
    }

    async placeTestOrder(symbol, side, size, price = null) {
        return this.request('POST', '/orders/test', { symbol, side, size, price });
    }

    async closePosition(positionId, confirmPhrase = null) {
        return this.request('POST', '/orders/close-position', { position_id: positionId, confirm_phrase: confirmPhrase });
    }

    async cancelAllOrders(confirmPhrase = null) {
        return this.request('POST', '/orders/cancel-all', { confirm_phrase: confirmPhrase });
    }

    // Strategies
    async getStrategies() {
        return this.request('GET', '/strategies');
    }

    async enableStrategy(strategyId, enabled) {
        return this.request('POST', `/strategies/${strategyId}/enable`, { enabled });
    }

    async updateStrategyParams(strategyId, params) {
        return this.request('POST', `/strategies/${strategyId}/params`, { params });
    }

    // Risk
    async getRiskStatus() {
        return this.request('GET', '/risk/status');
    }

    async getRiskLimits() {
        return this.request('GET', '/risk/limits');
    }

    async updateRiskLimits(limits) {
        return this.request('POST', '/risk/limits', limits);
    }

    // Analytics
    async getAnalyticsSummary() {
        return this.request('GET', '/analytics/summary');
    }

    async getBlockedSignals() {
        return this.request('GET', '/analytics/blocks');
    }

    async runBacktest(symbol, startDate, endDate, initialCapital = 10000) {
        return this.request('POST', '/analytics/backtest', { symbol, start_date: startDate, end_date: endDate, initial_capital: initialCapital });
    }

    async runEvaluate(symbol, days = 7) {
        return this.request('POST', '/analytics/evaluate', { symbol, days });
    }

    // Jobs
    async listJobs(limit = 100) {
        return this.request('GET', `/jobs?limit=${limit}`);
    }

    async getJob(jobId) {
        return this.request('GET', `/jobs/${jobId}`);
    }

    // System
    async reconcileNow() {
        return this.request('POST', '/reconcile/now');
    }

    async runSmokeTest() {
        return this.request('POST', '/smoke-test/run');
    }

    // SSE Streams
    createEventStream(callback) {
        const url = `${this.baseURL}/events/stream`;
        const eventSource = new EventSource(url);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                callback(data);
            } catch (error) {
                console.error('Failed to parse event:', error);
            }
        };

        eventSource.onerror = (error) => {
            console.error('Event stream error:', error);
            eventSource.close();
            // Reconnect after delay
            setTimeout(() => {
                this.createEventStream(callback);
            }, 5000);
        };

        return eventSource;
    }

    createLogStream(callback) {
        const url = `${this.baseURL}/logs/stream`;
        const eventSource = new EventSource(url);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                callback(data);
            } catch (error) {
                console.error('Failed to parse log:', error);
            }
        };

        eventSource.onerror = (error) => {
            console.error('Log stream error:', error);
            eventSource.close();
            // Reconnect after delay
            setTimeout(() => {
                this.createLogStream(callback);
            }, 5000);
        };

        return eventSource;
    }
}

// Global instance
const api = new HEANApiClient();

