/**
 * Eureka UI - Visual Singularity
 * 3D Force-Directed Graph with Energy Flow Visualization
 * Future Shadow: 500ms ahead price projection
 */

class EurekaVisualization {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.nodes = [];
        this.edges = [];
        this.energyFlows = [];
        this.currentPrices = new Map();
        this.futurePrices = new Map();
        this.leaderAsset = null;
        this.apiUrl = 'http://localhost:8000';
        
        // Animation state
        this.animationId = null;
        this.lastUpdateTime = 0;
        this.updateInterval = 100; // Update every 100ms
        
        // Price history for Future Shadow and Oracle View
        this.priceHistory = [];
        this.maxHistoryLength = 100;
        
        // Oracle predictions (500ms, 1s, 5s)
        this.oraclePredictions = {
            '500ms': [],
            '1s': [],
            '5s': []
        };
        this.currentOracleData = null;
        
        // Latency monitoring
        this.latencyHistory = [];
        this.maxLatencyHistory = 100;
        this.internalLatency = 0;
        this.jitter = 0;
        this.packetToOrderLatency = 0;
        this.latencyChartG = null;
        this.latencyChartWidth = 0;
        this.latencyChartHeight = 180;
        
        // System Evolution Level (SEL)
        this.systemEvolutionLevel = 0.0;
        this.selUpdateInterval = 2000; // Update every 2 seconds
        
        // Causal Web visualization
        this.causalWebCanvas = null;
        this.causalWebCtx = null;
        this.causalEdges = [];
        this.causalNodes = [];
        this.causalWebScene = null;
        this.causalWebCamera = null;
        this.causalWebRenderer = null;
        
        this.init();
    }
    
    async init() {
        // Initialize Three.js scene
        this.setupScene();
        
        // Initialize D3 for price chart
        this.setupPriceChart();
        
        // Initialize D3 for latency chart
        this.setupLatencyChart();
        
        // Initialize Causal Web visualization
        this.setupCausalWeb();
        
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        
        // Start data fetching
        await this.fetchGraphData();
        
        // Start animation loop
        this.animate();
        
        // Start periodic updates
        setInterval(() => this.updateData(), this.updateInterval);
        
        // Start latency monitoring updates
        setInterval(() => this.updateLatencyData(), 50); // Update every 50ms
        
        // Start SEL updates
        setInterval(() => this.updateSEL(), this.selUpdateInterval);
        
        // Start causal web updates
        setInterval(() => this.updateCausalWeb(), 1000); // Update every second
    }
    
    setupLatencyChart() {
        const chartContainer = d3.select('#latency-chart');
        chartContainer.selectAll('*').remove();
        
        const margin = { top: 10, right: 20, bottom: 30, left: 60 };
        this.latencyChartWidth = 510 - margin.left - margin.right;
        
        const svg = chartContainer.append('svg')
            .attr('width', this.latencyChartWidth + margin.left + margin.right)
            .attr('height', this.latencyChartHeight + margin.top + margin.bottom);
        
        this.latencyChartG = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
    }
    
    async updateLatencyData() {
        try {
            // Fetch latency metrics from API
            const response = await fetch(`${this.apiUrl}/api/latency/metrics`);
            if (response.ok) {
                const latencyData = await response.json();
                
                this.internalLatency = latencyData.internal_latency_us || 0;
                this.jitter = latencyData.jitter_us || 0;
                this.packetToOrderLatency = latencyData.packet_to_order_us || 0;
                
                // Update UI
                this.updateLatencyUI();
                
                // Add to history
                this.latencyHistory.push({
                    timestamp: Date.now(),
                    internal: this.internalLatency,
                    jitter: this.jitter,
                    packetToOrder: this.packetToOrderLatency
                });
                
                // Keep history size limited
                if (this.latencyHistory.length > this.maxLatencyHistory) {
                    this.latencyHistory.shift();
                }
                
                // Update chart
                this.updateLatencyChart();
            }
        } catch (e) {
            // Simulate latency data for demo
            this.internalLatency = Math.random() * 80 + 10; // 10-90 μs
            this.jitter = Math.random() * 20 + 5; // 5-25 μs
            this.packetToOrderLatency = Math.random() * 90 + 10; // 10-100 μs
            
            this.updateLatencyUI();
            
            this.latencyHistory.push({
                timestamp: Date.now(),
                internal: this.internalLatency,
                jitter: this.jitter,
                packetToOrder: this.packetToOrderLatency
            });
            
            if (this.latencyHistory.length > this.maxLatencyHistory) {
                this.latencyHistory.shift();
            }
            
            this.updateLatencyChart();
        }
    }
    
    updateLatencyUI() {
        const internalEl = document.getElementById('internal-latency');
        const jitterEl = document.getElementById('jitter');
        const packetToOrderEl = document.getElementById('packet-to-order');
        const targetStatusEl = document.getElementById('target-status');
        const statusEl = document.getElementById('latency-status');
        
        internalEl.textContent = `${this.internalLatency.toFixed(1)} μs`;
        jitterEl.textContent = `${this.jitter.toFixed(1)} μs`;
        packetToOrderEl.textContent = `${this.packetToOrderLatency.toFixed(1)} μs`;
        
        // Color coding based on latency thresholds
        const setColor = (el, value, goodThreshold = 50, warnThreshold = 80) => {
            el.className = 'latency-stat-value';
            if (value > warnThreshold) {
                el.classList.add('critical');
            } else if (value > goodThreshold) {
                el.classList.add('warning');
            }
        };
        
        setColor(internalEl, this.internalLatency);
        setColor(jitterEl, this.jitter, 15, 30);
        setColor(packetToOrderEl, this.packetToOrderLatency);
        
        // Overall status
        const maxLatency = Math.max(this.internalLatency, this.jitter, this.packetToOrderLatency);
        if (maxLatency < 50) {
            targetStatusEl.textContent = 'PASS';
            targetStatusEl.className = 'latency-stat-value';
            statusEl.textContent = '● OPTIMAL';
            statusEl.style.color = '#00ff88';
        } else if (maxLatency < 80) {
            targetStatusEl.textContent = 'WARN';
            targetStatusEl.className = 'latency-stat-value warning';
            statusEl.textContent = '● WARNING';
            statusEl.style.color = '#ffaa00';
        } else {
            targetStatusEl.textContent = 'FAIL';
            targetStatusEl.className = 'latency-stat-value critical';
            statusEl.textContent = '● CRITICAL';
            statusEl.style.color = '#ff4444';
        }
    }
    
    updateLatencyChart() {
        if (!this.latencyChartG || this.latencyHistory.length < 2) return;
        
        const data = this.latencyHistory.slice(-50);
        const margin = { top: 10, right: 20, bottom: 30, left: 60 };
        
        // Clear previous
        this.latencyChartG.selectAll('*').remove();
        
        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, data.length - 1])
            .range([0, this.latencyChartWidth]);
        
        const allLatencies = [
            ...data.map(d => d.internal),
            ...data.map(d => d.jitter),
            ...data.map(d => d.packetToOrder),
            100 // Target line
        ];
        
        const yScale = d3.scaleLinear()
            .domain([0, Math.max(150, d3.max(allLatencies) * 1.1)])
            .range([this.latencyChartHeight, 0]);
        
        // Target line (100 μs)
        const targetLine = d3.line()
            .x((d, i) => xScale(i))
            .y(() => yScale(100))
            .curve(d3.curveLinear);
        
        this.latencyChartG.append('path')
            .datum(data)
            .attr('class', 'latency-line target')
            .attr('d', targetLine);
        
        // Internal latency line
        const internalLine = d3.line()
            .x((d, i) => xScale(i))
            .y(d => yScale(d.internal))
            .curve(d3.curveMonotoneX);
        
        this.latencyChartG.append('path')
            .datum(data)
            .attr('class', 'latency-line internal')
            .attr('d', internalLine);
        
        // Jitter line
        const jitterLine = d3.line()
            .x((d, i) => xScale(i))
            .y(d => yScale(d.jitter))
            .curve(d3.curveMonotoneX);
        
        this.latencyChartG.append('path')
            .datum(data)
            .attr('class', 'latency-line jitter')
            .attr('d', jitterLine);
        
        // Axes
        const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat(() => '');
        const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat(d => `${d}μs`);
        
        this.latencyChartG.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${this.latencyChartHeight})`)
            .call(xAxis);
        
        this.latencyChartG.append('g')
            .attr('class', 'axis')
            .call(yAxis);
        
        // Legend
        const legend = this.latencyChartG.append('g')
            .attr('transform', `translate(${this.latencyChartWidth - 120}, 10)`);
        
        const legendData = [
            { label: 'Internal', color: '#00ff88' },
            { label: 'Jitter', color: '#ff00ff' },
            { label: 'Target 100μs', color: '#ff4444' }
        ];
        
        legendData.forEach((item, i) => {
            const itemG = legend.append('g')
                .attr('transform', `translate(0, ${i * 15})`);
            
            itemG.append('line')
                .attr('x1', 0)
                .attr('x2', 15)
                .attr('y1', 0)
                .attr('y2', 0)
                .attr('stroke', item.color)
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', item.label.includes('Target') ? '2,2' : '0');
            
            itemG.append('text')
                .attr('x', 20)
                .attr('y', 4)
                .attr('fill', '#fff')
                .attr('font-size', '10px')
                .text(item.label);
        });
    }
    
    setupScene() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e27);
        this.scene.fog = new THREE.FogExp2(0x0a0e27, 0.002);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            10000
        );
        this.camera.position.set(0, 0, 500);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        document.getElementById('canvas-container').appendChild(this.renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0x00d9ff, 0.8);
        directionalLight.position.set(100, 100, 100);
        this.scene.add(directionalLight);
        
        const pointLight = new THREE.PointLight(0xff00ff, 0.5);
        pointLight.position.set(-100, -100, 100);
        this.scene.add(pointLight);
        
        // Controls (simple orbit)
        this.setupControls();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    setupControls() {
        let isMouseDown = false;
        let mouseX = 0, mouseY = 0;
        let cameraAngleX = 0, cameraAngleY = 0;
        
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            isMouseDown = true;
            mouseX = e.clientX;
            mouseY = e.clientY;
        });
        
        this.renderer.domElement.addEventListener('mousemove', (e) => {
            if (isMouseDown) {
                const deltaX = e.clientX - mouseX;
                const deltaY = e.clientY - mouseY;
                
                cameraAngleY += deltaX * 0.005;
                cameraAngleX += deltaY * 0.005;
                
                const radius = 500;
                this.camera.position.x = radius * Math.sin(cameraAngleY) * Math.cos(cameraAngleX);
                this.camera.position.y = radius * Math.sin(cameraAngleX);
                this.camera.position.z = radius * Math.cos(cameraAngleY) * Math.cos(cameraAngleX);
                this.camera.lookAt(0, 0, 0);
                
                mouseX = e.clientX;
                mouseY = e.clientY;
            }
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            isMouseDown = false;
        });
        
        // Zoom with wheel
        this.renderer.domElement.addEventListener('wheel', (e) => {
            const delta = e.deltaY * 0.01;
            this.camera.position.multiplyScalar(1 + delta);
            this.camera.position.clampLength(100, 2000);
        });
    }
    
    setupPriceChart() {
        const margin = { top: 10, right: 20, bottom: 30, left: 50 };
        const width = document.getElementById('price-chart').offsetWidth - margin.left - margin.right;
        const height = 150 - margin.top - margin.bottom;
        
        const svg = d3.select('#price-chart')
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        this.priceChartSvg = svg;
        this.priceChartG = g;
        this.priceChartWidth = width;
        this.priceChartHeight = height;
    }
    
    async fetchGraphData() {
        try {
            // Fetch graph engine data from API
            const response = await fetch(`${this.apiUrl}/api/graph-engine/state`);
            if (!response.ok) {
                throw new Error('Failed to fetch graph data');
            }
            
            const data = await response.json();
            this.updateGraph(data);
            
            // Fetch leader info
            const leaderResponse = await fetch(`${this.apiUrl}/api/graph-engine/leader`);
            if (leaderResponse.ok) {
                const leaderData = await leaderResponse.json();
                this.leaderAsset = leaderData.leader;
                this.updateLeaderDisplay(leaderData);
            }
        } catch (error) {
            console.error('Error fetching graph data:', error);
            // Use mock data for demonstration
            this.useMockData();
        }
    }
    
    useMockData() {
        // Mock data for demonstration
        const mockAssets = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT'
        ];
        
        const mockData = {
            assets: mockAssets.map(symbol => ({
                symbol,
                price: Math.random() * 100000,
                correlation: {},
                leader_score: Math.random()
            })),
            correlations: {},
            leader: 'BTCUSDT'
        };
        
        // Generate correlations
        mockAssets.forEach(a => {
            mockAssets.forEach(b => {
                if (a !== b) {
                    mockData.correlations[`${a}-${b}`] = Math.random() * 2 - 1;
                }
            });
        });
        
        this.updateGraph(mockData);
        this.leaderAsset = 'BTCUSDT';
    }
    
    updateGraph(data) {
        // Clear existing nodes and edges
        this.nodes.forEach(node => this.scene.remove(node.mesh));
        this.edges.forEach(edge => this.scene.remove(edge.line));
        this.energyFlows.forEach(flow => this.scene.remove(flow));
        
        this.nodes = [];
        this.edges = [];
        this.energyFlows = [];
        
        // Create nodes (assets)
        const assetCount = data.assets.length;
        const radius = 200;
        
        data.assets.forEach((asset, index) => {
            const angle = (index / assetCount) * Math.PI * 2;
            const x = radius * Math.cos(angle);
            const y = radius * Math.sin(angle);
            const z = (Math.random() - 0.5) * 100;
            
            // Determine node color based on leader status
            let color = 0x888888; // Neutral
            if (asset.symbol === this.leaderAsset) {
                color = 0x00d9ff; // Leader (cyan)
            } else if (asset.leader_score < 0) {
                color = 0xff6b6b; // Laggard (red)
            }
            
            // Create sphere geometry for node
            const geometry = new THREE.SphereGeometry(10, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color,
                emissive: color,
                emissiveIntensity: 0.3,
                transparent: true,
                opacity: 0.9
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(x, y, z);
            mesh.userData = { symbol: asset.symbol, asset };
            
            // Add label (text sprite would be better, but using simple approach)
            const label = this.createLabel(asset.symbol, x, y, z + 15);
            
            this.scene.add(mesh);
            this.scene.add(label);
            
            this.nodes.push({
                mesh,
                label,
                symbol: asset.symbol,
                position: new THREE.Vector3(x, y, z),
                velocity: new THREE.Vector3(0, 0, 0),
                asset
            });
            
            // Store price
            this.currentPrices.set(asset.symbol, asset.price);
        });
        
        // Create edges (correlations) and energy flows
        this.nodes.forEach(nodeA => {
            this.nodes.forEach(nodeB => {
                if (nodeA.symbol === nodeB.symbol) return;
                
                const correlation = data.correlations[`${nodeA.symbol}-${nodeB.symbol}`] || 0;
                
                if (Math.abs(correlation) > 0.5) {
                    // Create edge
                    const geometry = new THREE.BufferGeometry().setFromPoints([
                        nodeA.position,
                        nodeB.position
                    ]);
                    
                    const material = new THREE.LineBasicMaterial({
                        color: 0x444444,
                        transparent: true,
                        opacity: Math.abs(correlation) * 0.3
                    });
                    
                    const line = new THREE.Line(geometry, material);
                    this.scene.add(line);
                    
                    this.edges.push({
                        line,
                        nodeA,
                        nodeB,
                        correlation
                    });
                    
                    // Create energy flow if leader -> laggard
                    if (nodeA.symbol === this.leaderAsset && correlation > 0.7) {
                        this.createEnergyFlow(nodeA, nodeB);
                    }
                }
            });
        });
        
        // Update UI
        document.getElementById('asset-count').textContent = assetCount;
    }
    
    createLabel(text, x, y, z) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 128;
        canvas.height = 64;
        
        context.fillStyle = '#ffffff';
        context.font = '12px Arial';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, 64, 32);
        
        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.position.set(x, y, z);
        sprite.scale.set(50, 25, 1);
        
        return sprite;
    }
    
    createEnergyFlow(fromNode, toNode) {
        const particleCount = 10;
        const particles = [];
        
        for (let i = 0; i < particleCount; i++) {
            const geometry = new THREE.SphereGeometry(2, 8, 8);
            const material = new THREE.MeshBasicMaterial({
                color: 0xff00ff,
                transparent: true,
                opacity: 0.8
            });
            
            const particle = new THREE.Mesh(geometry, material);
            particle.position.copy(fromNode.position);
            particle.userData = {
                progress: i / particleCount,
                from: fromNode.position,
                to: toNode.position
            };
            
            this.scene.add(particle);
            particles.push(particle);
        }
        
        this.energyFlows.push({
            particles,
            fromNode,
            toNode
        });
    }
    
    updateEnergyFlows() {
        this.energyFlows.forEach(flow => {
            flow.particles.forEach(particle => {
                particle.userData.progress += 0.02;
                if (particle.userData.progress > 1) {
                    particle.userData.progress = 0;
                }
                
                const from = particle.userData.from;
                const to = particle.userData.to;
                const progress = particle.userData.progress;
                
                particle.position.lerpVectors(from, to, progress);
                
                // Pulsing effect
                const scale = 1 + Math.sin(Date.now() * 0.005 + progress * 10) * 0.5;
                particle.scale.set(scale, scale, scale);
            });
        });
    }
    
    updateLeaderDisplay(data) {
        if (data.leader) {
            document.getElementById('leader-name').textContent = data.leader;
            document.getElementById('leader-score').textContent = 
                `Leader Score: ${data.leader_score?.toFixed(2) || '0.00'}`;
        }
    }
    
    updatePriceChart() {
        if (!this.priceChartG || this.priceHistory.length < 2) return;
        
        const data = this.priceHistory.slice(-50); // Last 50 points
        const margin = { top: 10, right: 20, bottom: 30, left: 50 };
        
        // Clear previous
        this.priceChartG.selectAll('*').remove();
        
        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, data.length - 1])
            .range([0, this.priceChartWidth]);
        
        const prices = data.map(d => d.current);
        const futurePrices = data.map(d => d.future);
        const allPrices = [...prices, ...futurePrices].filter(p => p != null);
        
        const yScale = d3.scaleLinear()
            .domain(d3.extent(allPrices))
            .range([this.priceChartHeight, 0]);
        
        // Current price line
        const currentLine = d3.line()
            .x((d, i) => xScale(i))
            .y(d => yScale(d.current))
            .curve(d3.curveMonotoneX);
        
        this.priceChartG.append('path')
            .datum(data)
            .attr('class', 'price-line current')
            .attr('d', currentLine);
        
        // Future shadow line (500ms ahead)
        const futureLine = d3.line()
            .x((d, i) => xScale(i))
            .y(d => yScale(d.future))
            .curve(d3.curveMonotoneX);
        
        this.priceChartG.append('path')
            .datum(data)
            .attr('class', 'price-line future')
            .attr('d', futureLine);
        
        // Axes
        const xAxis = d3.axisBottom(xScale).ticks(5);
        const yAxis = d3.axisLeft(yScale).ticks(5);
        
        this.priceChartG.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${this.priceChartHeight})`)
            .call(xAxis);
        
        this.priceChartG.append('g')
            .attr('class', 'axis')
            .call(yAxis);
    }
    
    async updateData() {
        try {
            // Fetch Oracle predictions from API
            const response = await fetch(`${this.apiUrl}/api/oracle/predictions`);
            if (response.ok) {
                const oracleData = await response.json();
                if (oracleData && oracleData.price_predictions) {
                    this.currentOracleData = oracleData;
                }
            }
        } catch (e) {
            // Fallback to simulation if API unavailable
        }
        
        // Simulate price updates and future predictions
        this.nodes.forEach(node => {
            const currentPrice = this.currentPrices.get(node.symbol) || 0;
            
            // Simulate price movement
            const newPrice = currentPrice * (1 + (Math.random() - 0.5) * 0.001);
            this.currentPrices.set(node.symbol, newPrice);
            
            // Use Oracle predictions if available, otherwise simulate
            let futurePrice = newPrice * (1 + (Math.random() - 0.5) * 0.0005);
            if (this.currentOracleData && this.currentOracleData.price_predictions) {
                const preds = this.currentOracleData.price_predictions;
                futurePrice = preds['500ms'] ? preds['500ms'].price : futurePrice;
            }
            
            this.futurePrices.set(node.symbol, futurePrice);
            
            // Update price history
            this.priceHistory.push({
                timestamp: Date.now(),
                current: newPrice,
                future: futurePrice,
                future500ms: this.currentOracleData && this.currentOracleData.price_predictions ? 
                            this.currentOracleData.price_predictions['500ms']?.price : futurePrice,
                future1s: this.currentOracleData && this.currentOracleData.price_predictions ? 
                         this.currentOracleData.price_predictions['1s']?.price : futurePrice,
                future5s: this.currentOracleData && this.currentOracleData.price_predictions ? 
                         this.currentOracleData.price_predictions['5s']?.price : futurePrice
            });
            
            if (this.priceHistory.length > this.maxHistoryLength) {
                this.priceHistory.shift();
            }
        });
        
        // Update price chart
        this.updatePriceChart();
        
        // Calculate energy intensity
        const energyIntensity = this.energyFlows.length * 0.1;
        document.getElementById('energy-intensity').textContent = energyIntensity.toFixed(2);
        
        // Update prediction stats if Oracle data available
        if (this.currentOracleData) {
            const tcnProb = (this.currentOracleData.tcn_reversal_prob * 100).toFixed(1);
            const avgConf = this.currentOracleData.price_predictions ? 
                ((this.currentOracleData.price_predictions['500ms']?.confidence * 100 || 0).toFixed(0)) : '0';
            const statsEl = document.getElementById('prediction-stats');
            if (statsEl) {
                statsEl.textContent = `Confidence: ${avgConf}% | Reversal Prob: ${tcnProb}%`;
            }
        }
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        // Update energy flows
        this.updateEnergyFlows();
        
        // Apply force-directed layout (simplified)
        this.applyForces();
        
        // Render
        this.renderer.render(this.scene, this.camera);
    }
    
    applyForces() {
        // Simplified force-directed layout
        this.nodes.forEach(node => {
            const force = new THREE.Vector3(0, 0, 0);
            
            // Repulsion from other nodes
            this.nodes.forEach(otherNode => {
                if (node === otherNode) return;
                
                const direction = new THREE.Vector3()
                    .subVectors(node.position, otherNode.position);
                const distance = direction.length();
                
                if (distance > 0) {
                    direction.normalize();
                    force.add(direction.multiplyScalar(100 / (distance * distance)));
                }
            });
            
            // Apply force
            node.velocity.add(force.multiplyScalar(0.01));
            node.velocity.multiplyScalar(0.9); // Damping
            node.position.add(node.velocity);
        });
        
        // Update edge positions
        this.edges.forEach(edge => {
            const geometry = edge.line.geometry;
            const positions = geometry.attributes.position.array;
            positions[0] = edge.nodeA.position.x;
            positions[1] = edge.nodeA.position.y;
            positions[2] = edge.nodeA.position.z;
            positions[3] = edge.nodeB.position.x;
            positions[4] = edge.nodeB.position.y;
            positions[5] = edge.nodeB.position.z;
            geometry.attributes.position.needsUpdate = true;
        });
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    async updateSEL() {
        try {
            const response = await fetch(`${this.apiUrl}/api/metamorphic/sel`);
            if (response.ok) {
                const data = await response.json();
                this.systemEvolutionLevel = data.sel || 0.0;
                this.updateSELDisplay();
            }
        } catch (error) {
            // Use mock data for demo
            this.systemEvolutionLevel = Math.min(1.0, this.systemEvolutionLevel + (Math.random() - 0.5) * 0.01);
            this.updateSELDisplay();
        }
    }
    
    updateSELDisplay() {
        const selEl = document.getElementById('system-evolution-level');
        const descEl = document.getElementById('sel-description');
        const causalEdgesEl = document.getElementById('causal-edges');
        
        if (selEl) {
            selEl.textContent = this.systemEvolutionLevel.toFixed(3);
            selEl.className = 'stat sel-level';
            
            // Color coding based on SEL level
            if (this.systemEvolutionLevel < 0.2) {
                selEl.classList.add('novice');
                if (descEl) descEl.textContent = 'Novice';
            } else if (this.systemEvolutionLevel < 0.4) {
                selEl.classList.add('intermediate');
                if (descEl) descEl.textContent = 'Intermediate';
            } else if (this.systemEvolutionLevel < 0.6) {
                selEl.classList.add('advanced');
                if (descEl) descEl.textContent = 'Advanced';
            } else if (this.systemEvolutionLevel < 0.8) {
                selEl.classList.add('expert');
                if (descEl) descEl.textContent = 'Expert';
            } else if (this.systemEvolutionLevel < 0.95) {
                selEl.classList.add('master');
                if (descEl) descEl.textContent = 'Master';
            } else {
                selEl.classList.add('singularity');
                if (descEl) descEl.textContent = 'SINGULARITY';
            }
        }
        
        // Update causal edges count
        if (causalEdgesEl) {
            causalEdgesEl.textContent = this.causalEdges.length;
        }
    }
    
    setupCausalWeb() {
        const canvas = document.getElementById('causal-web-canvas');
        if (!canvas) return;
        
        this.causalWebCanvas = canvas;
        this.causalWebCtx = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = 360;
        canvas.height = 330;
        
        // Initialize Three.js scene for 3D causal web
        this.causalWebScene = new THREE.Scene();
        this.causalWebCamera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
        this.causalWebRenderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
        this.causalWebRenderer.setSize(canvas.width, canvas.height);
        this.causalWebRenderer.setClearColor(0x000000, 0);
        
        this.causalWebCamera.position.z = 5;
        
        // Add lighting
        const light = new THREE.AmbientLight(0x404040);
        this.causalWebScene.add(light);
        const directionalLight = new THREE.DirectionalLight(0x00d9ff, 0.8);
        directionalLight.position.set(1, 1, 1);
        this.causalWebScene.add(directionalLight);
    }
    
    async updateCausalWeb() {
        try {
            const response = await fetch(`${this.apiUrl}/api/causal/graph`);
            if (response.ok) {
                const data = await response.json();
                this.causalEdges = data.edges || [];
                this.causalNodes = data.nodes || [];
                this.renderCausalWeb();
            }
        } catch (error) {
            // Use mock data for demo
            this.generateMockCausalData();
            this.renderCausalWeb();
        }
    }
    
    generateMockCausalData() {
        const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT'];
        this.causalNodes = symbols.map((symbol, i) => ({
            id: symbol,
            x: Math.cos((i / symbols.length) * Math.PI * 2) * 2,
            y: Math.sin((i / symbols.length) * Math.PI * 2) * 2,
            z: (Math.random() - 0.5) * 0.5
        }));
        
        this.causalEdges = [];
        for (let i = 0; i < symbols.length; i++) {
            for (let j = i + 1; j < symbols.length; j++) {
                if (Math.random() > 0.6) {
                    this.causalEdges.push({
                        source: symbols[i],
                        target: symbols[j],
                        strength: Math.random(),
                        lag_us: Math.floor(Math.random() * 1000000)
                    });
                }
            }
        }
    }
    
    renderCausalWeb() {
        if (!this.causalWebScene || !this.causalWebRenderer) return;
        
        // Clear previous objects
        while(this.causalWebScene.children.length > 2) { // Keep lights
            this.causalWebScene.remove(this.causalWebScene.children[2]);
        }
        
        // Create node map
        const nodeMap = new Map();
        this.causalNodes.forEach(node => {
            nodeMap.set(node.id, node);
        });
        
        // Render edges (causal relationships)
        this.causalEdges.forEach(edge => {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            
            if (source && target) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(source.x, source.y, source.z),
                    new THREE.Vector3(target.x, target.y, target.z)
                ]);
                
                const material = new THREE.LineBasicMaterial({
                    color: 0x00d9ff,
                    opacity: edge.strength,
                    transparent: true
                });
                
                const line = new THREE.Line(geometry, material);
                this.causalWebScene.add(line);
            }
        });
        
        // Render nodes (assets)
        this.causalNodes.forEach(node => {
            const geometry = new THREE.SphereGeometry(0.1, 16, 16);
            const material = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(node.x, node.y, node.z);
            this.causalWebScene.add(sphere);
        });
        
        // Rotate camera for animation
        this.causalWebCamera.position.x = Math.sin(Date.now() / 5000) * 3;
        this.causalWebCamera.position.y = Math.cos(Date.now() / 5000) * 3;
        this.causalWebCamera.lookAt(0, 0, 0);
        
        this.causalWebRenderer.render(this.causalWebScene, this.causalWebCamera);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new EurekaVisualization();
});
