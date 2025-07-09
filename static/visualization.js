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