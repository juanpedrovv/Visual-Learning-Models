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
        this.isPlayingTrajectory = false;
        
        // New progressive animation properties
        this.trajectoryAnimationId = null;
        this.currentTrajectoryBatch = 0;
        this.trajectoriesPerBatch = 8; // Show only 8 trajectories at a time
        this.trajectoryDuration = 2000; // Duration for each trajectory animation
        this.trajectoryFadeTime = 1000; // Time for trajectory to fade out
        this.maxVisibleTrajectories = 15; // Maximum trajectories visible simultaneously
        
        // Visualization dimensions
        this.margin = { top: 20, right: 20, bottom: 40, left: 40 };
        this.width = 500 - this.margin.left - this.margin.right;
        this.height = 360 - this.margin.bottom - this.margin.top;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadDatasets();
        this.createTooltip();
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
        
        document.getElementById('animate-epochs').addEventListener('click', () => {
            this.animateEpochs();
        });
        
        document.getElementById('animate-layers').addEventListener('click', () => {
            this.animateLayers();
        });
        
        document.getElementById('play-trajectories').addEventListener('click', () => {
            this.playTrajectoryAnimation();
        });
        
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
        
        // Epoch and layer selection
        document.getElementById('epoch-select').addEventListener('change', (e) => {
            this.currentEpochIndex = parseInt(e.target.value);
            this.updateVisualization();
        });
        
        document.getElementById('layer-select').addEventListener('change', (e) => {
            this.currentLayerIndex = parseInt(e.target.value);
            this.updateVisualization();
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
    }
    
    createEpochVisualization() {
        const container = document.getElementById('epoch-viz');
        container.innerHTML = '';
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom);
        
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
            .attr('y', this.height + 35)
            .style('text-anchor', 'middle')
            .text('Dimension 1');
        
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -this.height / 2)
            .attr('y', -25)
            .style('text-anchor', 'middle')
            .text('Dimension 2');
        
        this.epochSvg = svg;
        this.updateEpochVisualization();
    }
    
    createLayerVisualization() {
        const container = document.getElementById('layer-viz');
        container.innerHTML = '';
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom);
        
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
            .attr('y', this.height + 35)
            .style('text-anchor', 'middle')
            .text('Dimension 1');
        
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -this.height / 2)
            .attr('y', -25)
            .style('text-anchor', 'middle')
            .text('Dimension 2');
        
        this.layerSvg = svg;
        this.updateLayerVisualization();
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
    }
    
    drawScatterplot(g, data, coordsKey, type) {
        const xScale = type === 'epoch' ? this.epochXScale : this.layerXScale;
        const yScale = type === 'epoch' ? this.epochYScale : this.layerYScale;
        
        // Remove existing circles
        g.selectAll('.data-point').remove();
        
        // Add circles
        const circles = g.selectAll('.data-point')
            .data(data[coordsKey])
            .enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('cx', d => xScale(d[0]))
            .attr('cy', d => yScale(d[1]))
            .attr('r', 3)
            .attr('fill', (d, i) => this.colorScale(data.labels[i]))
            .attr('opacity', 0.7)
            .style('cursor', 'pointer');
        
        // Add interactivity
        circles
            .on('mouseover', (event, d, i) => {
                const index = circles.nodes().indexOf(event.target);
                const label = data.labels[index];
                const coords = data[coordsKey][index];
                
                this.tooltip.transition()
                    .duration(200)
                    .style('opacity', 0.9);
                
                this.tooltip.html(`
                    <strong>Class:</strong> ${label}<br/>
                    <strong>Coordinates:</strong> (${coords[0].toFixed(2)}, ${coords[1].toFixed(2)})<br/>
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
            .on('click', (event, d, i) => {
                const index = circles.nodes().indexOf(event.target);
                const label = data.labels[index];
                console.log(`Clicked on point: class ${label}, epoch ${data.epoch}, layer ${data.layer}`);
        
                // Could add more click functionality here
                this.highlightSameClass(label);
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
            console.log(`Starting progressive animation with ${trajectoryData.length} trajectories`);
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
                trajectories.push({
                    points: trajectoryPoints,
                    class: trajectoryPoints[0].label,
                    color: this.colorScale(trajectoryPoints[0].label),
                    sampleIdx: sampleIdx
                });
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
        const color = trajectoryData.color;
        
        // Create path element
        const path = g.append('path')
            .datum(pathData)
            .attr('class', 'trajectory-path')
            .attr('d', line)
            .attr('stroke', color)
            .attr('stroke-width', 2)
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
            .attr('stroke-opacity', 0.6);
        
        // Add start and end points
        const startPoint = g.append('circle')
            .attr('class', 'trajectory-point start')
            .attr('cx', xScale(trajectoryData.points[0].coords[0]))
            .attr('cy', yScale(trajectoryData.points[0].coords[1]))
            .attr('r', 0)
            .attr('fill', color)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('pointer-events', 'none');
        
        const endPoint = g.append('circle')
            .attr('class', 'trajectory-point end')
            .attr('cx', xScale(trajectoryData.points[trajectoryData.points.length - 1].coords[0]))
            .attr('cy', yScale(trajectoryData.points[trajectoryData.points.length - 1].coords[1]))
            .attr('r', 0)
            .attr('fill', color)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('pointer-events', 'none');
        
        // Animate points
        startPoint.transition()
            .delay(200)
            .duration(400)
            .attr('r', 3);
            
        endPoint.transition()
            .delay(this.trajectoryDuration * 0.6)
            .duration(400)
            .attr('r', 4);
        
        // Add arrow at end if enabled
        let arrow = null;
        if (this.showArrows && trajectoryData.points.length > 2) {
            arrow = this.addTrajectoryArrow(g, trajectoryData.points, xScale, yScale, color);
        }
        
        return { path, startPoint, endPoint, arrow };
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
    
    playTrajectoryAnimation() {
        if (!this.currentData) return;
        
        this.isPlayingTrajectory = true;
        const button = document.getElementById('play-trajectories');
        button.textContent = 'Stop Journey';
        button.disabled = false;
        
        // Animate through all epochs and layers
        let epochIndex = 0;
        let layerIndex = 0;
        
        const animate = () => {
            if (!this.isPlayingTrajectory) {
                button.textContent = 'Play Full Journey';
                return;
            }
            
            this.currentEpochIndex = epochIndex;
            this.currentLayerIndex = layerIndex;
            
            this.updateVisualization();
            
            // Progress through epochs first, then layers
            epochIndex++;
            if (epochIndex >= this.currentData.epochs.length) {
                epochIndex = 0;
                layerIndex++;
                if (layerIndex >= this.currentData.layers.length) {
                    // Animation complete
                    this.isPlayingTrajectory = false;
                    button.textContent = 'Play Full Journey';
                    return;
                }
            }
            
            setTimeout(animate, this.animationSpeed / 2);
        };
        
        // Toggle animation
        if (this.isPlayingTrajectory) {
            this.isPlayingTrajectory = false;
            button.textContent = 'Play Full Journey';
        } else {
            animate();
        }
    }
    
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
        
        // Create legend for the 10 classes (0-9)
        for (let i = 0; i < 10; i++) {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            
            const colorBox = document.createElement('div');
            colorBox.className = 'legend-color';
            colorBox.style.backgroundColor = this.colorScale(i);
            
            const label = document.createElement('span');
            label.textContent = `Class ${i}`;
            
            legendItem.appendChild(colorBox);
            legendItem.appendChild(label);
            legendContainer.appendChild(legendItem);
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
            const progress = (this.currentEpochIndex / (this.currentData.epochs.length - 1)) * 100;
            document.getElementById('epoch-progress').style.width = progress + '%';
            
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
            const progress = (this.currentLayerIndex / (this.currentData.layers.length - 1)) * 100;
            document.getElementById('layer-progress').style.width = progress + '%';
            
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
        if (!this.currentData) {
            this.showError('Please load a dataset first');
            return;
        }
        
        const button = document.getElementById('play-progressive-epoch');
        
        // If already running, stop it
        if (this.trajectoryAnimationId) {
            this.clearTrajectoryAnimation();
            button.textContent = '🎬 Progressive';
            button.classList.remove('stopping');
            return;
        }
        
        // Force trajectory display and start progressive animation for epoch visualization
        button.textContent = '⏹️ Stop Progressive';
        button.classList.add('stopping');
        
        // Ensure trajectories are enabled
        document.getElementById('show-trajectories').checked = true;
        this.showTrajectories = true;
        
        // Clear any existing animation and start progressive animation for epoch
        this.clearTrajectoryAnimation();
        
        // Get current epoch data for progressive animation
        const currentLayer = this.currentData.layers[this.currentLayerIndex];
        if (!currentLayer) return;
        
        const reductionMethod = document.querySelector('input[name="reduction"]:checked').value;
        const coordsKey = reductionMethod === 'tsne' ? 'tsne_coords' : 'umap_coords';
        
        const epochData = this.currentData.epochs.map(epoch => {
            const key = `epoch_${epoch}_layer_${currentLayer}`;
            return this.currentData.projections[key];
        }).filter(d => d);
        
        if (epochData.length > 1) {
            const g = this.epochSvg.select('g');
            this.drawProgressiveTrajectories(g, epochData, coordsKey, 'epoch', this.epochXScale, this.epochYScale);
        }
        
        // Store reference to reset button after animation completes
        const resetButton = () => {
            button.textContent = '🎬 Progressive';
            button.classList.remove('stopping');
        };
        
        // Reset button after 15 seconds or when animation ends
        setTimeout(resetButton, 15000);
    }
    
    playProgressiveLayerAnimation() {
        if (!this.currentData) {
            this.showError('Please load a dataset first');
            return;
        }
        
        const button = document.getElementById('play-progressive-layer');
        
        // If already running, stop it
        if (this.trajectoryAnimationId) {
            this.clearTrajectoryAnimation();
            button.textContent = '🎬 Progressive';
            button.classList.remove('stopping');
            return;
        }
        
        // Force trajectory display and start progressive animation for layer visualization
        button.textContent = '⏹️ Stop Progressive';
        button.classList.add('stopping');
        
        // Ensure trajectories are enabled
        document.getElementById('show-trajectories').checked = true;
        this.showTrajectories = true;
        
        // Clear any existing animation and start progressive animation for layer
        this.clearTrajectoryAnimation();
        
        // Get current layer data for progressive animation
        const currentEpoch = this.currentData.epochs[this.currentEpochIndex];
        if (currentEpoch === undefined) return;
        
        const reductionMethod = document.querySelector('input[name="reduction"]:checked').value;
        const coordsKey = reductionMethod === 'tsne' ? 'tsne_coords' : 'umap_coords';
        
        const layerData = this.currentData.layers.map(layer => {
            const key = `epoch_${currentEpoch}_layer_${layer}`;
            return this.currentData.projections[key];
        }).filter(d => d);
        
        if (layerData.length > 1) {
            const g = this.layerSvg.select('g');
            this.drawProgressiveTrajectories(g, layerData, coordsKey, 'layer', this.layerXScale, this.layerYScale);
        }
        
        // Store reference to reset button after animation completes
        const resetButton = () => {
            button.textContent = '🎬 Progressive';
            button.classList.remove('stopping');
        };
        
        // Reset button after 15 seconds or when animation ends
        setTimeout(resetButton, 15000);
    }
    
    animateEpochs() {
        if (!this.currentData) return;
        
        // Create a smooth transition through all epochs
        const epochs = this.currentData.epochs;
        let currentIndex = 0;
        
        const animate = () => {
            if (currentIndex >= epochs.length) return;
            
            this.currentEpochIndex = currentIndex;
            this.updateEpochVisualization();
            
            currentIndex++;
            setTimeout(animate, this.animationSpeed);
        };
        
        animate();
    }
    
    animateLayers() {
        if (!this.currentData) return;
        
        // Create a smooth transition through all layers
        const layers = this.currentData.layers;
        let currentIndex = 0;
        
        const animate = () => {
            if (currentIndex >= layers.length) return;
            
            this.currentLayerIndex = currentIndex;
            this.updateLayerVisualization();
            
            currentIndex++;
            setTimeout(animate, this.animationSpeed);
        };
        
        animate();
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