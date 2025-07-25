// Neural Network Visualization with D3.js
class NeuralNetworkVisualization {
    constructor() {
        this.currentDataset = null;
        this.currentData = null;
        this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        this.isAnimating = false;
        this.animationSpeed = 1000; // ms between frames
        this.currentEpochIndex = 0;
        this.currentLayerIndex = 0;
        this.showTrajectories = true;
        this.showArrows = false;
        this.trajectoryAnimationIndex = 0;
        
        // Class filtering system
        this.activeClasses = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); // All classes active by default
        
        // New progressive animation properties
        this.trajectoryAnimationId = null;
        this.currentTrajectoryBatch = 0;
        this.trajectoriesPerBatch = 8; // Show only 8 trajectories at a time
        this.trajectoryDuration = 2000; // Duration for each trajectory animation
        this.trajectoryFadeTime = 1000; // Time for trajectory to fade out
        this.maxVisibleTrajectories = 15; // Maximum trajectories visible simultaneously
        
        // Visualization dimensions - responsive
        this.margin = { top: 20, right: 20, bottom: 50, left: 50 };
        this.updateDimensions();
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadDatasets();
        this.createTooltip();
        
        // Add resize listener for responsive design
        window.addEventListener('resize', () => {
            this.updateDimensions();
            if (this.epochSvg) this.resizeVisualization('epoch');
            if (this.layerSvg) this.resizeVisualization('layer');
        });
    }
    
    updateDimensions() {
        // Get container dimensions dynamically
        const container = document.querySelector('.viz-content');
        if (container) {
            const containerWidth = container.clientWidth;
            const containerHeight = container.clientHeight;
            
            // Calculate optimal dimensions with some padding
            this.width = Math.max(350, Math.min(600, containerWidth - this.margin.left - this.margin.right - 20));
            this.height = Math.max(280, Math.min(450, containerHeight - this.margin.top - this.margin.bottom - 20));
        } else {
            // Fallback dimensions
            this.width = 400;
            this.height = 320;
        }
    }
    
    setupEventListeners() {
        // Dataset selection
        document.getElementById('dataset-select').addEventListener('change', (e) => {
            this.onDatasetChange(e.target.value);
        });
        
        // Control buttons
        document.getElementById('load-data').addEventListener('click', () => {
            this.loadVisualization();
        });
        
        // Removed animate-epochs, animate-layers, and play-trajectories buttons
        
        // Trajectory controls
        document.getElementById('show-trajectories').addEventListener('change', (e) => {
            this.showTrajectories = e.target.checked;
            this.updateVisualization();
        });
        
        document.getElementById('show-arrows').addEventListener('change', (e) => {
            this.showArrows = e.target.checked;
            this.updateVisualization();
        });
        
        // Animation controls
        document.getElementById('play-epoch').addEventListener('click', () => {
            this.playEpochAnimation();
        });
        
        document.getElementById('pause-epoch').addEventListener('click', () => {
            this.pauseEpochAnimation();
        });
        
        document.getElementById('play-layer').addEventListener('click', () => {
            this.playLayerAnimation();
        });
        
        document.getElementById('pause-layer').addEventListener('click', () => {
            this.pauseLayerAnimation();
        });
        
        // Progressive animation buttons
        document.getElementById('play-progressive-epoch').addEventListener('click', () => {
            this.playProgressiveEpochAnimation();
        });
        
        document.getElementById('play-progressive-layer').addEventListener('click', () => {
            this.playProgressiveLayerAnimation();
        });
        
        // Next buttons
        document.getElementById('next-epoch').addEventListener('click', () => {
            this.nextEpoch();
        });
        
        document.getElementById('next-layer').addEventListener('click', () => {
            this.nextLayer();
        });
        
        // Epoch and layer selection
        document.getElementById('epoch-select').addEventListener('change', (e) => {
            this.currentEpochIndex = parseInt(e.target.value);
            this.updateVisualization();
            this.updateProgressBars();
        });
        
        document.getElementById('layer-select').addEventListener('change', (e) => {
            this.currentLayerIndex = parseInt(e.target.value);
            this.updateVisualization();
            this.updateProgressBars();
        });
        
        // Reduction method change
        document.querySelectorAll('input[name="reduction"]').forEach(radio => {
            radio.addEventListener('change', () => {
                this.updateVisualization();
            });
        });
        
        // Animation settings
        document.getElementById('animation-speed').addEventListener('change', (e) => {
            this.trajectoryDuration = parseInt(e.target.value);
        });
        
        document.getElementById('trajectory-density').addEventListener('change', (e) => {
            this.trajectoriesPerBatch = parseInt(e.target.value);
        });
        

    }
    
    async loadDatasets() {
        try {
            const response = await fetch('/api/datasets');
            const datasets = await response.json();
            
            const select = document.getElementById('dataset-select');
            select.innerHTML = '<option value="">Select a dataset...</option>';
            
            datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset.name;
                option.textContent = `${dataset.name.toUpperCase()} (${dataset.summary.total_projections} projections)`;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading datasets:', error);
            this.showError('Failed to load datasets');
        }
    }
    
    async onDatasetChange(datasetName) {
        if (!datasetName) return;
        
        this.currentDataset = datasetName;
        
        try {
            const response = await fetch(`/api/data/${datasetName}`);
            this.currentData = await response.json();
            
            this.populateControls();
            this.updateDatasetInfo();
        } catch (error) {
            console.error('Error loading dataset:', error);
            this.showError(`Failed to load dataset: ${datasetName}`);
        }
    }
    
    populateControls() {
        if (!this.currentData) return;
        
        // Populate epoch select
        const epochSelect = document.getElementById('epoch-select');
        epochSelect.innerHTML = '<option value="">Select epoch...</option>';
        this.currentData.epochs.forEach((epoch, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `Epoch ${epoch}`;
            epochSelect.appendChild(option);
        });
        epochSelect.disabled = false;
        
        // Populate layer select
        const layerSelect = document.getElementById('layer-select');
        layerSelect.innerHTML = '<option value="">Select layer...</option>';
        this.currentData.layers.forEach((layer, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = layer;
            layerSelect.appendChild(option);
        });
        layerSelect.disabled = false;
    }
    
    updateDatasetInfo() {
        if (!this.currentData) return;
        
        document.getElementById('current-dataset').textContent = this.currentData.dataset;
        document.getElementById('available-epochs').textContent = this.currentData.epochs.join(', ');
        document.getElementById('available-layers').textContent = this.currentData.layers.join(', ');
        
        // Get sample count from first available projection
        const firstProjection = Object.values(this.currentData.projections)[0];
        if (firstProjection) {
            document.getElementById('sample-count').textContent = firstProjection.n_samples;
        }
        
        document.getElementById('dataset-info').style.display = 'flex';
    }
    
    createTooltip() {
        this.tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);
    }
    
    loadVisualization() {
        if (!this.currentData) {
            this.showError('Please select a dataset first');
            return;
        }
        
        // Set default indices if not set
        if (this.currentEpochIndex === undefined || this.currentEpochIndex >= this.currentData.epochs.length) {
            this.currentEpochIndex = 0;
        }
        if (this.currentLayerIndex === undefined || this.currentLayerIndex >= this.currentData.layers.length) {
            this.currentLayerIndex = 0;
        }
        
        this.createEpochVisualization();
        this.createLayerVisualization();
        this.createLegend();
        this.updateProgressBars();
    }
    
    createEpochVisualization() {
        const container = document.getElementById('epoch-viz');
        container.innerHTML = '';
        
        // Update dimensions before creating visualization
        this.updateDimensions();
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom)
            .attr('viewBox', `0 0 ${this.width + this.margin.left + this.margin.right} ${this.height + this.margin.top + this.margin.bottom}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('max-width', '100%')
            .style('height', 'auto');
        
        const g = svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
        
        // Create scales
        this.epochXScale = d3.scaleLinear().range([0, this.width]);
        this.epochYScale = d3.scaleLinear().range([this.height, 0]);
        
        // Add axes
        g.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${this.height})`);
        
        g.append('g')
            .attr('class', 'y-axis');
        
        // Add axis labels
        g.append('text')
            .attr('class', 'axis-label')
            .attr('x', this.width / 2)
            .attr('y', this.height + Math.min(35, this.margin.bottom - 5))
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .text('Dimension 1');
        
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -this.height / 2)
            .attr('y', -Math.min(25, this.margin.left - 15))
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .text('Dimension 2');
        
        this.epochSvg = svg;
        this.updateEpochVisualization();
    }
    
    createLayerVisualization() {
        const container = document.getElementById('layer-viz');
        container.innerHTML = '';
        
        // Update dimensions before creating visualization
        this.updateDimensions();
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom)
            .attr('viewBox', `0 0 ${this.width + this.margin.left + this.margin.right} ${this.height + this.margin.top + this.margin.bottom}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('max-width', '100%')
            .style('height', 'auto');
        
        const g = svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
        
        // Create scales
        this.layerXScale = d3.scaleLinear().range([0, this.width]);
        this.layerYScale = d3.scaleLinear().range([this.height, 0]);
        
        // Add axes
        g.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${this.height})`);
        
        g.append('g')
            .attr('class', 'y-axis');
        
        // Add axis labels
        g.append('text')
            .attr('class', 'axis-label')
            .attr('x', this.width / 2)
            .attr('y', this.height + Math.min(35, this.margin.bottom - 5))
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .text('Dimension 1');
        
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -this.height / 2)
            .attr('y', -Math.min(25, this.margin.left - 15))
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .text('Dimension 2');
        
        this.layerSvg = svg;
        this.updateLayerVisualization();
    }
    
    resizeVisualization(type) {
        if (!this.currentData) return;
        
        this.updateDimensions();
        
        if (type === 'epoch' && this.epochSvg) {
            // Update SVG dimensions
            this.epochSvg
                .attr('width', this.width + this.margin.left + this.margin.right)
                .attr('height', this.height + this.margin.top + this.margin.bottom)
                .attr('viewBox', `0 0 ${this.width + this.margin.left + this.margin.right} ${this.height + this.margin.top + this.margin.bottom}`);
            
            // Update scales range
            this.epochXScale.range([0, this.width]);
            this.epochYScale.range([this.height, 0]);
            
            // Re-draw visualization
            this.updateEpochVisualization();
        }
        
        if (type === 'layer' && this.layerSvg) {
            // Update SVG dimensions
            this.layerSvg
                .attr('width', this.width + this.margin.left + this.margin.right)
                .attr('height', this.height + this.margin.top + this.margin.bottom)
                .attr('viewBox', `0 0 ${this.width + this.margin.left + this.margin.right} ${this.height + this.margin.top + this.margin.bottom}`);
            
            // Update scales range
            this.layerXScale.range([0, this.width]);
            this.layerYScale.range([this.height, 0]);
            
            // Re-draw visualization
            this.updateLayerVisualization();
        }
    }
    

    
    updateVisualization() {
        if (!this.currentData) return;
        
        // Clear any running trajectory animations before updating
        this.clearTrajectoryAnimation();
        
        this.updateEpochVisualization();
        this.updateLayerVisualization();
    }
    
    updateEpochVisualization() {
        if (!this.epochSvg || !this.currentData) return;
        
        const currentLayer = this.currentData.layers[this.currentLayerIndex];
        if (!currentLayer) return;
        
        const reductionMethod = document.querySelector('input[name="reduction"]:checked').value;
        const coordsKey = reductionMethod === 'tsne' ? 'tsne_coords' : 'umap_coords';
        
        // Get data for all epochs for the current layer
        const epochData = this.currentData.epochs.map(epoch => {
            const key = `epoch_${epoch}_layer_${currentLayer}`;
            return this.currentData.projections[key];
        }).filter(d => d);
        
        if (epochData.length === 0) return;
        
        // Combine all coordinates to set consistent scales
        const allCoords = epochData.flatMap(d => d[coordsKey]);
        const xExtent = d3.extent(allCoords, d => d[0]);
        const yExtent = d3.extent(allCoords, d => d[1]);
        
        this.epochXScale.domain(xExtent);
        this.epochYScale.domain(yExtent);
        
        // Update axes
        const g = this.epochSvg.select('g');
        g.select('.x-axis').call(d3.axisBottom(this.epochXScale));
        g.select('.y-axis').call(d3.axisLeft(this.epochYScale));
        
        // Only draw static trajectories if enabled and not in progressive mode
        if (this.showTrajectories && epochData.length > 1 && !this.trajectoryAnimationId) {
            this.drawStaticTrajectories(g, epochData, coordsKey, 'epoch', this.epochXScale, this.epochYScale);
        }
        
        // Show current epoch data
        const currentEpochData = epochData[this.currentEpochIndex];
        if (currentEpochData) {
            this.drawScatterplot(g, currentEpochData, coordsKey, 'epoch');
        }
        
        // Update epoch counter
        document.getElementById('epoch-counter').textContent = `${this.currentEpochIndex + 1}/${epochData.length}`;
        
        // Update active classes info
        this.updateActiveClassesInfo();
    }
    
    updateLayerVisualization() {
        if (!this.layerSvg || !this.currentData) return;
        
        const currentEpoch = this.currentData.epochs[this.currentEpochIndex];
        if (currentEpoch === undefined) return;
        
        const reductionMethod = document.querySelector('input[name="reduction"]:checked').value;
        const coordsKey = reductionMethod === 'tsne' ? 'tsne_coords' : 'umap_coords';
        
        // Get data for all layers for the current epoch
        const layerData = this.currentData.layers.map(layer => {
            const key = `epoch_${currentEpoch}_layer_${layer}`;
            return this.currentData.projections[key];
        }).filter(d => d);
        
        if (layerData.length === 0) return;
        
        // Combine all coordinates to set consistent scales
        const allCoords = layerData.flatMap(d => d[coordsKey]);
        const xExtent = d3.extent(allCoords, d => d[0]);
        const yExtent = d3.extent(allCoords, d => d[1]);
        
        this.layerXScale.domain(xExtent);
        this.layerYScale.domain(yExtent);
        
        // Update axes
        const g = this.layerSvg.select('g');
        g.select('.x-axis').call(d3.axisBottom(this.layerXScale));
        g.select('.y-axis').call(d3.axisLeft(this.layerYScale));
        
        // Only draw static trajectories if enabled and not in progressive mode
        if (this.showTrajectories && layerData.length > 1 && !this.trajectoryAnimationId) {
            this.drawStaticTrajectories(g, layerData, coordsKey, 'layer', this.layerXScale, this.layerYScale);
        }
        
        // Show current layer data
        const currentLayerData = layerData[this.currentLayerIndex];
        if (currentLayerData) {
            this.drawScatterplot(g, currentLayerData, coordsKey, 'layer');
        }
        
        // Update layer counter
        document.getElementById('layer-counter').textContent = `${this.currentLayerIndex + 1}/${layerData.length}`;
        
        // Update active classes info
        this.updateActiveClassesInfo();
    }
    
    drawScatterplot(g, data, coordsKey, type) {
        const xScale = type === 'epoch' ? this.epochXScale : this.layerXScale;
        const yScale = type === 'epoch' ? this.epochYScale : this.layerYScale;
        
        // Remove existing circles
        g.selectAll('.data-point').remove();
        
        // Filter data by active classes
        const filteredData = data[coordsKey].map((coord, i) => ({
            coord: coord,
            label: data.labels[i],
            index: i
        })).filter(item => this.activeClasses.has(item.label));
        
        // Add circles for filtered data
        const circles = g.selectAll('.data-point')
            .data(filteredData)
            .enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('cx', d => xScale(d.coord[0]))
            .attr('cy', d => yScale(d.coord[1]))
            .attr('r', 3)
            .attr('fill', d => this.colorScale(d.label))
            .attr('opacity', 0.7)
            .style('cursor', 'pointer');
        
        // Add interactivity
        circles
            .on('mouseover', (event, d) => {
                this.tooltip.transition()
                    .duration(200)
                    .style('opacity', 0.9);
                
                this.tooltip.html(`
                    <strong>Class:</strong> ${d.label}<br/>
                    <strong>Coordinates:</strong> (${d.coord[0].toFixed(2)}, ${d.coord[1].toFixed(2)})<br/>
                    <strong>Epoch:</strong> ${data.epoch}<br/>
                    <strong>Layer:</strong> ${data.layer}
                `)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
                
                // Highlight point
                d3.select(event.target)
                    .attr('r', 5)
                    .attr('stroke', '#000')
                    .attr('stroke-width', 2);
            })
            .on('mouseout', (event) => {
                this.tooltip.transition()
                    .duration(500)
                    .style('opacity', 0);
                
                // Reset point
                d3.select(event.target)
                    .attr('r', 3)
                    .attr('stroke', 'none');
            })
            .on('click', (event, d) => {
                console.log(`Clicked on point: class ${d.label}, epoch ${data.epoch}, layer ${data.layer}`);
        
                // Could add more click functionality here
                this.highlightSameClass(d.label);
            });
        
        // Animate entrance
        circles
            .attr('r', 0)
            .transition()
            .duration(500)
            .attr('r', 3);
    }
    
    drawStaticTrajectories(g, dataSequence, coordsKey, type, xScale, yScale) {
        // Remove existing trajectories
        g.selectAll('.trajectory-path').remove();
        g.selectAll('.trajectory-arrow').remove();
        g.selectAll('.trajectory-point').remove();
        
        if (dataSequence.length < 2) return;
        
        // Sample fewer points for cleaner display
        const sampleSize = Math.min(30, dataSequence[0][coordsKey].length);
        const sampleIndices = this.sampleIndices(dataSequence[0][coordsKey].length, sampleSize);
        
        // Create line generator
        const line = d3.line()
            .x(d => xScale(d[0]))
            .y(d => yScale(d[1]))
            .curve(d3.curveCardinal.tension(0.3));
        
        // Draw simple static trajectories
        sampleIndices.forEach(sampleIdx => {
            const trajectoryPoints = dataSequence.map(data => {
                if (sampleIdx < data[coordsKey].length) {
                    return {
                        coords: data[coordsKey][sampleIdx],
                        label: data.labels[sampleIdx],
                        step: type === 'epoch' ? data.epoch : data.layer
                    };
                }
                return null;
            }).filter(point => point !== null);
            
            if (trajectoryPoints.length < 2) return;
            
            const trajectoryClass = trajectoryPoints[0].label;
            
            // Only draw trajectory if class is active
            if (!this.activeClasses.has(trajectoryClass)) return;
            
            const trajectoryColor = this.colorScale(trajectoryClass);
            const pathData = trajectoryPoints.map(p => p.coords);
            
            // Draw simple path
            g.append('path')
                .datum(pathData)
                .attr('class', 'trajectory-path')
                .attr('d', line)
                .attr('stroke', trajectoryColor)
                .attr('stroke-width', 1.5)
                .attr('stroke-opacity', 0.3)
                .attr('fill', 'none')
                .style('pointer-events', 'none');
        });
    }
    
    drawProgressiveTrajectories(g, dataSequence, coordsKey, type, xScale, yScale) {
        // Clear any existing animation
        if (this.trajectoryAnimationId) {
            clearTimeout(this.trajectoryAnimationId);
        }
        
        // Remove existing trajectories
        g.selectAll('.trajectory-path').remove();
        g.selectAll('.trajectory-arrow').remove();
        g.selectAll('.trajectory-point').remove();
        
        if (dataSequence.length < 2) return;
        
        // Sample fewer points for cleaner animation
        const sampleSize = Math.min(50, dataSequence[0][coordsKey].length);
        const sampleIndices = this.sampleIndices(dataSequence[0][coordsKey].length, sampleSize);
        
        // Create line generator with smoother curves
        const line = d3.line()
            .x(d => xScale(d[0]))
            .y(d => yScale(d[1]))
            .curve(d3.curveCardinal.tension(0.4));
        
        // Prepare trajectory data grouped by class for smoother animation
        const trajectoryData = this.prepareTrajectoryData(sampleIndices, dataSequence, coordsKey, type);
        
        // Start progressive animation
        if (trajectoryData.length > 0) {
            const classChanges = trajectoryData.filter(t => t.hasClassChange).length;
            const transitionLabel = this.getTransitionLabel(type);
            console.log(`Starting next-step progressive animation: ${transitionLabel}`);
            console.log(`${trajectoryData.length} trajectories, ${classChanges} show class changes`);
            
            // Show class change statistics
            this.showClassChangeStats(trajectoryData, type);
        }
        this.animateProgressiveTrajectories(g, trajectoryData, line, xScale, yScale);
    }
    
    prepareTrajectoryData(sampleIndices, dataSequence, coordsKey, type) {
        const trajectories = [];
        
        sampleIndices.forEach(sampleIdx => {
            const trajectoryPoints = dataSequence.map(data => {
                if (sampleIdx < data[coordsKey].length) {
                    return {
                        coords: data[coordsKey][sampleIdx],
                        label: data.labels[sampleIdx],
                        step: type === 'epoch' ? data.epoch : data.layer
                    };
                }
                return null;
            }).filter(point => point !== null);
            
            if (trajectoryPoints.length >= 2) {
                const trajectoryClass = trajectoryPoints[0].label;
                
                // Only include trajectory if class is active
                if (this.activeClasses.has(trajectoryClass)) {
                    // Check if there's a class change along the trajectory
                    const startClass = trajectoryPoints[0].label;
                    const endClass = trajectoryPoints[trajectoryPoints.length - 1].label;
                    const hasClassChange = startClass !== endClass;
                    
                    trajectories.push({
                        points: trajectoryPoints,
                        class: trajectoryClass,
                        color: this.colorScale(trajectoryClass),
                        sampleIdx: sampleIdx,
                        hasClassChange: hasClassChange,
                        startClass: startClass,
                        endClass: endClass,
                        startColor: this.colorScale(startClass),
                        endColor: this.colorScale(endClass)
                    });
                }
            }
        });
        
        // Group trajectories by class for better visual flow
        const groupedByClass = {};
        trajectories.forEach(traj => {
            if (!groupedByClass[traj.class]) {
                groupedByClass[traj.class] = [];
            }
            groupedByClass[traj.class].push(traj);
        });
        
        // Shuffle within each class but keep classes together
        const shuffledTrajectories = [];
        Object.values(groupedByClass).forEach(classTrajectories => {
            // Shuffle trajectories within class
            for (let i = classTrajectories.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [classTrajectories[i], classTrajectories[j]] = [classTrajectories[j], classTrajectories[i]];
            }
            shuffledTrajectories.push(...classTrajectories.slice(0, 8)); // Limit per class
        });
        
        return shuffledTrajectories.slice(0, 40); // Overall limit
    }
    
    animateProgressiveTrajectories(g, trajectories, line, xScale, yScale) {
        if (trajectories.length === 0) return;
        
        let currentIndex = 0;
        const visibleTrajectories = new Map(); // Track visible trajectories for fading
        
        const animateNextBatch = () => {
            // Remove oldest trajectories if we have too many
            if (visibleTrajectories.size >= this.maxVisibleTrajectories) {
                const oldestKeys = Array.from(visibleTrajectories.keys()).slice(0, this.trajectoriesPerBatch);
                oldestKeys.forEach(key => {
                    const trajectory = visibleTrajectories.get(key);
                    this.fadeOutTrajectory(trajectory);
                    visibleTrajectories.delete(key);
                });
            }
            
            // Add new batch of trajectories
            const batchEnd = Math.min(currentIndex + this.trajectoriesPerBatch, trajectories.length);
            const currentBatch = trajectories.slice(currentIndex, batchEnd);
            
            currentBatch.forEach((trajectory, i) => {
                // Delay each trajectory in the batch slightly for staggered effect
                setTimeout(() => {
                    const animatedTrajectory = this.animateSingleTrajectory(g, trajectory, line, xScale, yScale);
                    visibleTrajectories.set(`traj_${currentIndex + i}`, animatedTrajectory);
                }, i * 150); // 150ms delay between trajectories in batch
            });
            
            currentIndex = batchEnd;
            
            // Continue animation if there are more trajectories
            if (currentIndex < trajectories.length) {
                this.trajectoryAnimationId = setTimeout(animateNextBatch, this.trajectoryDuration);
            } else {
                // Animation complete - start fading all remaining trajectories after a delay
                setTimeout(() => {
                    visibleTrajectories.forEach(trajectory => {
                        this.fadeOutTrajectory(trajectory);
                    });
                    visibleTrajectories.clear();
                }, 2000);
            }
        };
        
        animateNextBatch();
    }
    
    animateSingleTrajectory(g, trajectoryData, line, xScale, yScale) {
        const pathData = trajectoryData.points.map(p => p.coords);
        const hasClassChange = trajectoryData.hasClassChange;
        
        // Create path element with gradient if there's class change
        let pathColor, pathElement;
        
        if (hasClassChange) {
            // Create a linear gradient for class transition
            const gradientId = `gradient-${trajectoryData.sampleIdx}-${Date.now()}`;
            const defs = g.append('defs');
            const gradient = defs.append('linearGradient')
                .attr('id', gradientId)
                .attr('x1', '0%').attr('y1', '0%')
                .attr('x2', '100%').attr('y2', '0%');
            
            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', trajectoryData.startColor);
            
            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', trajectoryData.endColor);
            
            pathColor = `url(#${gradientId})`;
        } else {
            pathColor = trajectoryData.color;
        }
        
        // Create path element
        const path = g.append('path')
            .datum(pathData)
            .attr('class', 'trajectory-path')
            .attr('d', line)
            .attr('stroke', pathColor)
            .attr('stroke-width', hasClassChange ? 2.5 : 2)
            .attr('stroke-opacity', 0)
            .attr('fill', 'none')
            .style('pointer-events', 'none');
        
        // Get path length for animation
        const totalLength = path.node().getTotalLength();
        
        // Set up path for animation
        path
            .attr('stroke-dasharray', totalLength + ' ' + totalLength)
            .attr('stroke-dashoffset', totalLength);
        
        // Animate path drawing
        path.transition()
            .duration(this.trajectoryDuration * 0.7)
            .ease(d3.easeQuadInOut)
            .attr('stroke-dashoffset', 0)
            .attr('stroke-opacity', hasClassChange ? 0.8 : 0.6);
        
        // Add start point with original class color
        const startPoint = g.append('circle')
            .attr('class', 'trajectory-point start')
            .attr('cx', xScale(trajectoryData.points[0].coords[0]))
            .attr('cy', yScale(trajectoryData.points[0].coords[1]))
            .attr('r', 0)
            .attr('fill', trajectoryData.startColor)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('pointer-events', 'none');
        
        // Add end point with final class color (this is the key change!)
        const endPoint = g.append('circle')
            .attr('class', 'trajectory-point end')
            .attr('cx', xScale(trajectoryData.points[trajectoryData.points.length - 1].coords[0]))
            .attr('cy', yScale(trajectoryData.points[trajectoryData.points.length - 1].coords[1]))
            .attr('r', 0)
            .attr('fill', trajectoryData.endColor) // Use end class color!
            .attr('stroke', '#fff')
            .attr('stroke-width', hasClassChange ? 3 : 2) // Thicker border for class changes
            .style('pointer-events', 'none');
        
        // Add class change indicator if there's a change
        let changeIndicator = null;
        if (hasClassChange) {
            changeIndicator = g.append('circle')
                .attr('class', 'trajectory-change-indicator')
                .attr('cx', xScale(trajectoryData.points[trajectoryData.points.length - 1].coords[0]))
                .attr('cy', yScale(trajectoryData.points[trajectoryData.points.length - 1].coords[1]))
                .attr('r', 0)
                .attr('fill', 'none')
                .attr('stroke', trajectoryData.endColor)
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '3,3')
                .style('pointer-events', 'none');
        }
        
        // Animate points
        startPoint.transition()
            .delay(200)
            .duration(400)
            .attr('r', 3);
            
        endPoint.transition()
            .delay(this.trajectoryDuration * 0.6)
            .duration(400)
            .attr('r', hasClassChange ? 5 : 4); // Larger for class changes
        
        // Animate change indicator
        if (changeIndicator) {
            changeIndicator.transition()
                .delay(this.trajectoryDuration * 0.8)
                .duration(600)
                .attr('r', 8)
                .attr('stroke-opacity', 0.6);
        }
        
        // Add arrow at end if enabled (use end color)
        let arrow = null;
        if (this.showArrows && trajectoryData.points.length > 2) {
            arrow = this.addTrajectoryArrow(g, trajectoryData.points, xScale, yScale, trajectoryData.endColor);
        }
        
        return { path, startPoint, endPoint, arrow, changeIndicator };
    }
    
    fadeOutTrajectory(trajectory) {
        const duration = this.trajectoryFadeTime;
        
        if (trajectory.path) {
            trajectory.path.transition()
                .duration(duration)
                .attr('stroke-opacity', 0)
                .remove();
        }
        
        if (trajectory.startPoint) {
            trajectory.startPoint.transition()
                .duration(duration)
                .attr('r', 0)
                .attr('opacity', 0)
                .remove();
        }
        
        if (trajectory.endPoint) {
            trajectory.endPoint.transition()
                .duration(duration)
                .attr('r', 0)
                .attr('opacity', 0)
                .remove();
        }
        
        if (trajectory.arrow) {
            trajectory.arrow.transition()
                .duration(duration)
                .attr('opacity', 0)
                .remove();
        }
        
        if (trajectory.changeIndicator) {
            trajectory.changeIndicator.transition()
                .duration(duration)
                .attr('r', 0)
                .attr('stroke-opacity', 0)
                .remove();
        }
    }
    
    addTrajectoryArrow(g, trajectoryPoints, xScale, yScale, color) {
        if (trajectoryPoints.length < 2) return null;
        
        const endIdx = trajectoryPoints.length - 1;
        const current = trajectoryPoints[endIdx];
        const previous = trajectoryPoints[endIdx - 1];
        
        const dx = xScale(current.coords[0]) - xScale(previous.coords[0]);
        const dy = yScale(current.coords[1]) - yScale(previous.coords[1]);
        const angle = Math.atan2(dy, dx);
        
        const arrowSize = 8;
        
        const arrow = g.append('polygon')
            .attr('class', 'trajectory-arrow')
            .attr('points', `0,${-arrowSize/2} ${arrowSize},0 0,${arrowSize/2}`)
            .attr('transform', 
                `translate(${xScale(current.coords[0])},${yScale(current.coords[1])}) rotate(${angle * 180 / Math.PI})`)
            .attr('fill', color)
            .attr('opacity', 0)
            .style('pointer-events', 'none');
        
        // Animate arrow appearance
        arrow.transition()
            .delay(this.trajectoryDuration * 0.8)
            .duration(400)
            .attr('opacity', 0.8);
        
        return arrow;
    }
    
    // Clean up trajectories when changing visualization
    clearTrajectoryAnimation() {
        if (this.trajectoryAnimationId) {
            clearTimeout(this.trajectoryAnimationId);
            this.trajectoryAnimationId = null;
        }
    }
    
    sampleIndices(totalLength, sampleSize) {
        if (totalLength <= sampleSize) {
            return Array.from({length: totalLength}, (_, i) => i);
        }
        
        // Stratified sampling to get representative points
        const step = totalLength / sampleSize;
        const indices = [];
        for (let i = 0; i < sampleSize; i++) {
            indices.push(Math.floor(i * step));
        }
        return indices;
    }
    
    // Removed playTrajectoryAnimation() function - no longer needed
    
    highlightSameClass(targetClass) {
        // Highlight all points with the same class across all visualizations
        d3.selectAll('.data-point')
            .transition()
            .duration(300)
            .attr('opacity', (d, i) => {
                const circles = d3.selectAll('.data-point').nodes();
                const index = circles.indexOf(d3.select(this).node());
                // This is a simplified version - in a real implementation,
                // we'd need to track the data more carefully
                return 0.7;
            })
            .attr('stroke', (d, i) => {
                // Similar logic for highlighting
                return null;
            });
    }
    

    
    createLegend() {
        const legendContainer = document.getElementById('legend');
        legendContainer.innerHTML = '';
        
        // Add title for legend
        const title = document.createElement('div');
        title.className = 'legend-title';
        title.textContent = 'Filter by Class (click to toggle):';
        legendContainer.appendChild(title);
        
        // Create interactive legend buttons for the 10 classes (0-9)
        for (let i = 0; i < 10; i++) {
            const legendButton = document.createElement('button');
            legendButton.className = 'legend-button';
            legendButton.setAttribute('data-class', i);
            
            // Set initial active state
            if (this.activeClasses.has(i)) {
                legendButton.classList.add('active');
            }
            
            const colorBox = document.createElement('div');
            colorBox.className = 'legend-color';
            colorBox.style.backgroundColor = this.colorScale(i);
            
            const label = document.createElement('span');
            label.textContent = `Class ${i}`;
            
            legendButton.appendChild(colorBox);
            legendButton.appendChild(label);
            
            // Add click event for toggling
            legendButton.addEventListener('click', () => {
                this.toggleClass(i);
            });
            
            legendContainer.appendChild(legendButton);
        }
        
        // Add control buttons
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'legend-controls';
        
        const selectAllBtn = document.createElement('button');
        selectAllBtn.textContent = 'Select All';
        selectAllBtn.className = 'legend-control-btn';
        selectAllBtn.addEventListener('click', () => this.selectAllClasses());
        
        const selectNoneBtn = document.createElement('button');
        selectNoneBtn.textContent = 'Select None';
        selectNoneBtn.className = 'legend-control-btn';
        selectNoneBtn.addEventListener('click', () => this.selectNoneClasses());
        
        controlsDiv.appendChild(selectAllBtn);
        controlsDiv.appendChild(selectNoneBtn);
        legendContainer.appendChild(controlsDiv);
    }
    
    toggleClass(classIndex) {
        const button = document.querySelector(`[data-class="${classIndex}"]`);
        
        if (this.activeClasses.has(classIndex)) {
            // Deactivate class
            this.activeClasses.delete(classIndex);
            button.classList.remove('active');
        } else {
            // Activate class
            this.activeClasses.add(classIndex);
            button.classList.add('active');
        }
        
        // Update visualizations
        this.updateVisualization();
        
        console.log(`Toggled class ${classIndex}. Active classes:`, Array.from(this.activeClasses));
    }
    
    selectAllClasses() {
        this.activeClasses = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        
        // Update button states
        document.querySelectorAll('.legend-button').forEach(button => {
            button.classList.add('active');
        });
        
        this.updateVisualization();
        console.log('All classes selected');
    }
    
    selectNoneClasses() {
        this.activeClasses.clear();
        
        // Update button states
        document.querySelectorAll('.legend-button').forEach(button => {
            button.classList.remove('active');
        });
        
        this.updateVisualization();
        console.log('No classes selected');
    }
    
    updateActiveClassesInfo() {
        const activeCount = this.activeClasses.size;
        const totalCount = 10;
        
        // Find or create info element
        let infoElement = document.getElementById('active-classes-info');
        if (!infoElement) {
            infoElement = document.createElement('div');
            infoElement.id = 'active-classes-info';
            infoElement.className = 'active-classes-info';
            
            // Insert after dataset info
            const datasetInfo = document.getElementById('dataset-info');
            if (datasetInfo) {
                datasetInfo.parentNode.insertBefore(infoElement, datasetInfo.nextSibling);
            }
        }
        
        if (activeCount === 0) {
            infoElement.innerHTML = `
                <div style="color: #dc3545; font-weight: bold;">
                    ⚠️ No classes selected - No data will be displayed
                </div>
            `;
        } else if (activeCount === totalCount) {
            infoElement.innerHTML = `
                <div style="color: #28a745;">
                    ✓ All classes active (${activeCount}/${totalCount})
                </div>
            `;
        } else {
            const activeClassesList = Array.from(this.activeClasses).sort().join(', ');
            infoElement.innerHTML = `
                <div style="color: #007bff;">
                    🎯 Filtered classes (${activeCount}/${totalCount}): ${activeClassesList}
                </div>
            `;
        }
    }
    
    showClassChangeStats(trajectoryData, type) {
        const classChanges = trajectoryData.filter(t => t.hasClassChange);
        const totalTrajectories = trajectoryData.length;
        
        if (classChanges.length === 0) return;
        
        // Group changes by transition type
        const changeStats = {};
        classChanges.forEach(traj => {
            const transition = `${traj.startClass} → ${traj.endClass}`;
            changeStats[transition] = (changeStats[transition] || 0) + 1;
        });
        
        // Create or update stats display
        let statsElement = document.getElementById('class-change-stats');
        if (!statsElement) {
            statsElement = document.createElement('div');
            statsElement.id = 'class-change-stats';
            statsElement.className = 'class-change-stats';
            
            // Insert after active classes info
            const activeClassesInfo = document.getElementById('active-classes-info');
            if (activeClassesInfo) {
                activeClassesInfo.parentNode.insertBefore(statsElement, activeClassesInfo.nextSibling);
            }
        }
        
        const changePercent = ((classChanges.length / totalTrajectories) * 100).toFixed(1);
        const typeLabel = type === 'epoch' ? 'epochs' : 'layers';
        
        // Determine transition direction
        const transitionLabel = this.getTransitionLabel(type);
        
        let statsHtml = `
            <div class="stats-header">
                🔄 Class Changes: ${transitionLabel}
            </div>
            <div class="stats-summary">
                ${classChanges.length}/${totalTrajectories} points (${changePercent}%) changed classification
            </div>
        `;
        
        if (Object.keys(changeStats).length > 0) {
            statsHtml += '<div class="stats-details">';
            Object.entries(changeStats)
                .sort(([,a], [,b]) => b - a) // Sort by frequency
                .slice(0, 5) // Show top 5
                .forEach(([transition, count]) => {
                    const [start, end] = transition.split(' → ');
                    statsHtml += `
                        <span class="transition-stat">
                            <span class="class-chip" style="background-color: ${this.colorScale(parseInt(start))}">${start}</span>
                            →
                            <span class="class-chip" style="background-color: ${this.colorScale(parseInt(end))}">${end}</span>
                            (${count})
                        </span>
                    `;
                });
            statsHtml += '</div>';
        }
        
        statsElement.innerHTML = statsHtml;
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            if (statsElement) {
                statsElement.style.opacity = '0.5';
            }
        }, 10000);
    }
    
    getTransitionLabel(type) {
        if (type === 'epoch') {
            const currentEpoch = this.currentData.epochs[this.currentEpochIndex];
            const nextEpochIndex = (this.currentEpochIndex + 1) % this.currentData.epochs.length;
            const nextEpoch = this.currentData.epochs[nextEpochIndex];
            const currentLayer = this.currentData.layers[this.currentLayerIndex];
            return `Epoch ${currentEpoch} → ${nextEpoch} (${currentLayer})`;
        } else {
            const currentEpoch = this.currentData.epochs[this.currentEpochIndex];
            const currentLayer = this.currentData.layers[this.currentLayerIndex];
            const nextLayerIndex = (this.currentLayerIndex + 1) % this.currentData.layers.length;
            const nextLayer = this.currentData.layers[nextLayerIndex];
            return `${currentLayer} → ${nextLayer} (Epoch ${currentEpoch})`;
        }
    }
    
    // Animation methods
    playEpochAnimation() {
        if (!this.currentData) return;
        
        this.isAnimating = true;
        document.getElementById('play-epoch').disabled = true;
        document.getElementById('pause-epoch').disabled = false;
        
        const animate = () => {
            if (!this.isAnimating) return;
            
            this.currentEpochIndex = (this.currentEpochIndex + 1) % this.currentData.epochs.length;
            this.updateEpochVisualization();
            
            // Update progress bar
            this.updateProgressBars();
            
            setTimeout(animate, this.animationSpeed);
        };
        
        animate();
    }
    
    pauseEpochAnimation() {
        this.isAnimating = false;
        document.getElementById('play-epoch').disabled = false;
        document.getElementById('pause-epoch').disabled = true;
    }
    
    playLayerAnimation() {
        if (!this.currentData) return;
        
        this.isAnimating = true;
        document.getElementById('play-layer').disabled = true;
        document.getElementById('pause-layer').disabled = false;
        
        const animate = () => {
            if (!this.isAnimating) return;
            
            this.currentLayerIndex = (this.currentLayerIndex + 1) % this.currentData.layers.length;
            this.updateLayerVisualization();
            
            // Update progress bar
            this.updateProgressBars();
            
            setTimeout(animate, this.animationSpeed);
        };
        
        animate();
    }
    
    pauseLayerAnimation() {
        this.isAnimating = false;
        document.getElementById('play-layer').disabled = false;
        document.getElementById('pause-layer').disabled = true;
    }
    
    playProgressiveEpochAnimation() {
        console.log('Progressive Epoch Animation button clicked');
        
        if (!this.currentData) {
            this.showError('Please load a dataset first');
            return;
        }
        
        const button = document.getElementById('play-progressive-epoch');
        
        // If already running, stop it
        if (this.trajectoryAnimationId) {
            this.clearTrajectoryAnimation();
            button.textContent = '🎬 Next Transition';
            button.classList.remove('stopping');
            return;
        }
        
        // Get transition info first for button text
        const currentLayer = this.currentData.layers[this.currentLayerIndex];
        if (!currentLayer) return;
        
        const nextEpochIndex = (this.currentEpochIndex + 1) % this.currentData.epochs.length;
        const currentEpoch = this.currentData.epochs[this.currentEpochIndex];
        const nextEpoch = this.currentData.epochs[nextEpochIndex];
        
        // Force trajectory display and start progressive animation for epoch visualization
        button.textContent = `⏹️ Stop (${currentEpoch}→${nextEpoch})`;
        button.classList.add('stopping');
        
        // Ensure trajectories are enabled
        document.getElementById('show-trajectories').checked = true;
        this.showTrajectories = true;
        
        // Clear any existing animation and start progressive animation for epoch
        this.clearTrajectoryAnimation();
        
        const reductionMethod = document.querySelector('input[name="reduction"]:checked').value;
        const coordsKey = reductionMethod === 'tsne' ? 'tsne_coords' : 'umap_coords';
        
        // Get current and next epoch data
        const currentKey = `epoch_${currentEpoch}_layer_${currentLayer}`;
        const nextKey = `epoch_${nextEpoch}_layer_${currentLayer}`;
        
        const currentData = this.currentData.projections[currentKey];
        const nextData = this.currentData.projections[nextKey];
        
        console.log(`Epoch transition: ${currentKey} → ${nextKey}`);
        console.log('Current data:', currentData ? 'Found' : 'Not found');
        console.log('Next data:', nextData ? 'Found' : 'Not found');
        
        if (currentData && nextData) {
            const transitionData = [currentData, nextData];
            const g = this.epochSvg.select('g');
            this.drawProgressiveTrajectories(g, transitionData, coordsKey, 'epoch', this.epochXScale, this.epochYScale);
        } else {
            console.error('Missing data for epoch transition');
            this.showError(`Cannot find data for transition ${currentEpoch}→${nextEpoch} in layer ${currentLayer}`);
        }
        
        // Store reference to reset button after animation completes
        const resetButton = () => {
            button.textContent = '🎬 Next Transition';
            button.classList.remove('stopping');
        };
        
        // Reset button after 15 seconds or when animation ends
        setTimeout(resetButton, 15000);
    }
    
    playProgressiveLayerAnimation() {
        console.log('Progressive Layer Animation button clicked');
        
        if (!this.currentData) {
            this.showError('Please load a dataset first');
            return;
        }
        
        const button = document.getElementById('play-progressive-layer');
        
        // If already running, stop it
        if (this.trajectoryAnimationId) {
            this.clearTrajectoryAnimation();
            button.textContent = '🎬 Next Transition';
            button.classList.remove('stopping');
            return;
        }
        
        // Get transition info first for button text
        const currentEpoch = this.currentData.epochs[this.currentEpochIndex];
        if (currentEpoch === undefined) return;
        
        const nextLayerIndex = (this.currentLayerIndex + 1) % this.currentData.layers.length;
        const currentLayer = this.currentData.layers[this.currentLayerIndex];
        const nextLayer = this.currentData.layers[nextLayerIndex];
        
        // Force trajectory display and start progressive animation for layer visualization
        button.textContent = `⏹️ Stop (${currentLayer}→${nextLayer})`;
        button.classList.add('stopping');
        
        // Ensure trajectories are enabled
        document.getElementById('show-trajectories').checked = true;
        this.showTrajectories = true;
        
        // Clear any existing animation and start progressive animation for layer
        this.clearTrajectoryAnimation();
        
        const reductionMethod = document.querySelector('input[name="reduction"]:checked').value;
        const coordsKey = reductionMethod === 'tsne' ? 'tsne_coords' : 'umap_coords';
        
        // Get current and next layer data
        const currentKey = `epoch_${currentEpoch}_layer_${currentLayer}`;
        const nextKey = `epoch_${currentEpoch}_layer_${nextLayer}`;
        
        const currentData = this.currentData.projections[currentKey];
        const nextData = this.currentData.projections[nextKey];
        
        console.log(`Layer transition: ${currentKey} → ${nextKey}`);
        console.log('Current data:', currentData ? 'Found' : 'Not found');
        console.log('Next data:', nextData ? 'Found' : 'Not found');
        
        if (currentData && nextData) {
            const transitionData = [currentData, nextData];
            const g = this.layerSvg.select('g');
            this.drawProgressiveTrajectories(g, transitionData, coordsKey, 'layer', this.layerXScale, this.layerYScale);
        } else {
            console.error('Missing data for layer transition');
            this.showError(`Cannot find data for transition ${currentLayer}→${nextLayer} in epoch ${currentEpoch}`);
        }
        
        // Store reference to reset button after animation completes
        const resetButton = () => {
            button.textContent = '🎬 Next Transition';
            button.classList.remove('stopping');
        };
        
        // Reset button after 15 seconds or when animation ends
        setTimeout(resetButton, 15000);
    }
    
    // Removed animateEpochs() and animateLayers() functions - no longer needed
    
    nextEpoch() {
        if (!this.currentData || !this.currentData.epochs) {
            this.showError('Please load a dataset first');
            return;
        }
        
        // Move to next epoch (cycle back to 0 if at end)
        this.currentEpochIndex = (this.currentEpochIndex + 1) % this.currentData.epochs.length;
        
        // Update epoch dropdown
        document.getElementById('epoch-select').value = this.currentEpochIndex;
        
        // Update both visualizations since epoch affects both panels
        this.updateVisualization();
        
        // Update progress bars
        this.updateProgressBars();
        
        console.log(`Advanced to epoch ${this.currentData.epochs[this.currentEpochIndex]} (index: ${this.currentEpochIndex})`);
    }
    
    nextLayer() {
        if (!this.currentData || !this.currentData.layers) {
            this.showError('Please load a dataset first');
            return;
        }
        
        // Move to next layer (cycle back to 0 if at end)
        this.currentLayerIndex = (this.currentLayerIndex + 1) % this.currentData.layers.length;
        
        // Update layer dropdown
        document.getElementById('layer-select').value = this.currentLayerIndex;
        
        // Update only layer visualization (epoch panel shouldn't change)
        this.updateLayerVisualization();
        
        // Update progress bars
        this.updateProgressBars();
        
        console.log(`Advanced to layer ${this.currentData.layers[this.currentLayerIndex]} (index: ${this.currentLayerIndex})`);
    }
    
    updateProgressBars() {
        if (!this.currentData) return;
        
        // Update epoch progress
        const epochProgress = (this.currentEpochIndex / (this.currentData.epochs.length - 1)) * 100;
        document.getElementById('epoch-progress').style.width = epochProgress + '%';
        
        // Update layer progress  
        const layerProgress = (this.currentLayerIndex / (this.currentData.layers.length - 1)) * 100;
        document.getElementById('layer-progress').style.width = layerProgress + '%';
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        
        document.querySelector('.container').insertBefore(errorDiv, document.querySelector('.control-panel'));
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Initialize the visualization when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new NeuralNetworkVisualization();
}); 