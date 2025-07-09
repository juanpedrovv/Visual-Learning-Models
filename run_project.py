#!/usr/bin/env python3
"""
Script de conveniencia para ejecutar el proyecto completo de visualización
de evolución de aprendizaje de redes neuronales.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def check_dependencies():
    """Verificar que las dependencias estén instaladas"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 'sklearn', 
        'pandas', 'flask', 'flask_cors', 'umap'
    ]
    
    print("🔍 Verificando dependencias...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Faltan las siguientes dependencias: {', '.join(missing_packages)}")
        print("Instálalas con: pip install -r requirements.txt")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def train_models():
    """Entrenar los modelos y generar datos de visualización"""
    print("\n🏋️  Iniciando entrenamiento de modelos...")
    print("⏳ Fases iniciales pueden tomar 5-15 minutos (descarga de datos, GPU warmup)")
    print("🚀 Una vez iniciado el entrenamiento será mucho más rápido")
    
    start_time = time.time()
    
    try:
        # Execute with real-time output instead of capturing
        print("\n" + "="*60)
        print("📝 SALIDA DEL ENTRENAMIENTO:")
        print("="*60)
        
        process = subprocess.Popen(
            [sys.executable, 'neural_network_trainer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                sys.stdout.flush()
        
        process.wait()
        
        if process.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"\n✅ Entrenamiento completado en {elapsed_time/60:.1f} minutos")
            return True
        else:
            print(f"\n❌ Error durante el entrenamiento (código: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando el entrenamiento: {e}")
        return False

def start_server():
    """Iniciar el servidor Flask"""
    print("\n🚀 Iniciando servidor Flask...")
    
    try:
        # Ejecutar el servidor en un proceso separado
        subprocess.run([sys.executable, 'server.py'])
    except KeyboardInterrupt:
        print("\n⛔ Servidor detenido por el usuario")
    except Exception as e:
        print(f"❌ Error iniciando el servidor: {e}")

def check_data_availability():
    """Verificar si los datos de visualización están disponibles"""
    data_dir = Path('visualization_data')
    
    if not data_dir.exists():
        return False, []
    
    datasets = []
    for file in data_dir.glob('*_summary.json'):
        dataset_name = file.stem.replace('_summary', '')
        datasets.append(dataset_name)
    
    return len(datasets) > 0, datasets

def main():
    parser = argparse.ArgumentParser(description='Ejecutar el proyecto de visualización de redes neuronales')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Saltar el entrenamiento de modelos')
    parser.add_argument('--only-server', action='store_true',
                       help='Solo iniciar el servidor (asume que los datos ya existen)')
    parser.add_argument('--check-deps', action='store_true',
                       help='Solo verificar las dependencias')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧠 NEURAL NETWORK LEARNING EVOLUTION VISUALIZATION")
    print("="*60)
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("\n✅ Verificación de dependencias completada")
        return
    
    # Verificar disponibilidad de datos
    data_available, datasets = check_data_availability()
    
    if args.only_server:
        if not data_available:
            print("❌ No se encontraron datos de visualización")
            print("Ejecuta primero el entrenamiento: python run_project.py")
            sys.exit(1)
        start_server()
        return
    
    # Entrenamiento de modelos
    if not args.skip_training:
        if data_available:
            print(f"\n📊 Datos encontrados para: {', '.join(datasets)}")
            response = input("¿Deseas re-entrenar los modelos? (y/N): ")
            if response.lower() not in ['y', 'yes', 'sí', 'si']:
                print("⏭️  Saltando entrenamiento")
            else:
                if not train_models():
                    sys.exit(1)
        else:
            if not train_models():
                sys.exit(1)
    else:
        if not data_available:
            print("❌ No se encontraron datos y se saltó el entrenamiento")
            print("Ejecuta sin --skip-training para generar los datos")
            sys.exit(1)
    
    # Verificar que los datos existan después del entrenamiento
    data_available, datasets = check_data_availability()
    if not data_available:
        print("❌ No se pudieron generar los datos de visualización")
        sys.exit(1)
    
    print(f"\n📊 Datos disponibles para: {', '.join(datasets)}")
    
    # Iniciar servidor
    print("\n" + "="*60)
    print("🌐 SERVIDOR LISTO")
    print("="*60)
    print("Abre tu navegador y ve a: http://localhost:5000")
    print("Presiona Ctrl+C para detener el servidor")
    print("="*60)
    
    start_server()

if __name__ == '__main__':
    main() 