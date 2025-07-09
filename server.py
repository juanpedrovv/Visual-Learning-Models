from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
DATA_DIR = "visualization_data"
TEMPLATE_DIR = "templates"
STATIC_DIR = "static"

@app.route('/')
def index():
    """Serve the main visualization page"""
    return render_template('index.html')

@app.route('/api/datasets')
def get_datasets():
    """Get list of available datasets"""
    try:
        datasets = []
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                if filename.endswith('_summary.json'):
                    dataset_name = filename.replace('_summary.json', '')
                    with open(os.path.join(DATA_DIR, filename), 'r') as f:
                        summary = json.load(f)
                        datasets.append({
                            'name': dataset_name,
                            'summary': summary
                        })
        return jsonify(datasets)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/<dataset_name>')
def get_dataset_data(dataset_name):
    """Get visualization data for a specific dataset"""
    try:
        filename = f"{dataset_name}_visualization_data.json"
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'Dataset {dataset_name} not found'}), 404
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projection/<dataset_name>/<epoch>/<layer>')
def get_projection_data(dataset_name, epoch, layer):
    """Get specific projection data for epoch and layer"""
    try:
        filename = f"{dataset_name}_visualization_data.json"
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'Dataset {dataset_name} not found'}), 404
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        key = f"epoch_{epoch}_layer_{layer}"
        if key not in data['projections']:
            return jsonify({'error': f'Projection for epoch {epoch} and layer {layer} not found'}), 404
        
        return jsonify(data['projections'][key])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare/<dataset_name>')
def get_comparison_data(dataset_name):
    """Get data for comparing different epochs/layers"""
    try:
        filename = f"{dataset_name}_visualization_data.json"
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'Dataset {dataset_name} not found'}), 404
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Prepare comparison data
        comparison_data = {
            'dataset': dataset_name,
            'epochs': data['epochs'],
            'layers': data['layers'],
            'epoch_evolution': {},
            'layer_evolution': {}
        }
        
        # Group by epochs (for T1: epoch evolution)
        for epoch in data['epochs']:
            comparison_data['epoch_evolution'][str(epoch)] = {}
            for layer in data['layers']:
                key = f"epoch_{epoch}_layer_{layer}"
                if key in data['projections']:
                    comparison_data['epoch_evolution'][str(epoch)][layer] = data['projections'][key]
        
        # Group by layers (for T2: layer evolution)
        for layer in data['layers']:
            comparison_data['layer_evolution'][layer] = {}
            for epoch in data['epochs']:
                key = f"epoch_{epoch}_layer_{layer}"
                if key in data['projections']:
                    comparison_data['layer_evolution'][layer][str(epoch)] = data['projections'][key]
        
        return jsonify(comparison_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(STATIC_DIR, filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_dir_exists': os.path.exists(DATA_DIR),
        'available_datasets': len([f for f in os.listdir(DATA_DIR) if f.endswith('_summary.json')]) if os.path.exists(DATA_DIR) else 0
    })

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    print("Starting Neural Network Visualization Server...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Server will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 