#!/usr/bin/env python3
"""
Script de entrenamiento para la red neuronal del controlador de TurtleSim
Genera datos sintéticos basados en reglas heurísticas y entrena el modelo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from turtle_nn_controller import NeuralNetworkController


def generate_training_data(num_samples=5000):
    """
    Genera datos de entrenamiento sintéticos basados en reglas heurísticas
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        # Generar entradas aleatorias
        dist_to_goal = np.random.uniform(0.1, 15.0)
        angle_to_goal = np.random.uniform(-math.pi, math.pi)
        dist_to_obstacle = np.random.uniform(0.1, 15.0)
        angle_to_obstacle = np.random.uniform(-math.pi, math.pi)
        current_linear_vel = np.random.uniform(0.0, 2.0)
        current_angular_vel = np.random.uniform(-3.0, 3.0)
        
        # Calcular salidas deseadas basadas en reglas heurísticas
        # Regla 1: Si hay obstáculo muy cerca, reducir velocidad y girar
        if dist_to_obstacle < 1.0:
            linear_vel = 0.2
            # Girar lejos del obstáculo
            if abs(angle_to_obstacle) < math.pi / 2:
                angular_vel = 2.5 * np.sign(angle_to_obstacle) if angle_to_obstacle != 0 else 2.5
            else:
                angular_vel = 1.5 * np.sign(angle_to_goal)
        # Regla 2: Si el obstáculo está cerca pero no crítico
        elif dist_to_obstacle < 2.5:
            linear_vel = 0.6
            # Combinar evitación y navegación
            if abs(angle_to_obstacle) < math.pi / 3:
                angular_vel = 1.8 * np.sign(angle_to_obstacle) if angle_to_obstacle != 0 else 1.8
            else:
                angular_vel = 1.2 * np.sign(angle_to_goal)
        # Regla 3: Sin obstáculos cercanos, navegar hacia el objetivo
        else:
            # Velocidad basada en distancia al objetivo
            if dist_to_goal < 1.0:
                linear_vel = 0.5
            elif dist_to_goal < 3.0:
                linear_vel = 1.2
            else:
                linear_vel = 1.5
            
            # Velocidad angular basada en error de ángulo
            angle_abs = abs(angle_to_goal)
            if angle_abs > 1.5:
                angular_vel = 2.0 * np.sign(angle_to_goal)
            elif angle_abs > 1.0:
                angular_vel = 1.5 * np.sign(angle_to_goal)
            elif angle_abs > 0.5:
                angular_vel = 1.0 * np.sign(angle_to_goal)
            else:
                angular_vel = 0.5 * np.sign(angle_to_goal)
        
        # Normalizar entradas
        inputs.append([
            min(dist_to_goal / 15.0, 1.0),
            angle_to_goal / math.pi,
            min(dist_to_obstacle / 15.0, 1.0),
            angle_to_obstacle / math.pi if dist_to_obstacle < 15.0 else 0.0,
            current_linear_vel / 2.0,
            current_angular_vel / 3.0
        ])
        
        # Targets NO normalizados (la red ya produce salidas en el rango correcto)
        targets.append([
            linear_vel,   # [0, 2.0]
            angular_vel   # [-3.0, 3.0]
        ])
    
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_model(model, inputs, targets, epochs=100, batch_size=32, learning_rate=0.001):
    """
    Entrena el modelo con los datos proporcionados
    """
    device = torch.device('cpu')
    model = model.to(device)
    model.train()
    
    # Convertir a tensores
    inputs_tensor = torch.FloatTensor(inputs).to(device)
    targets_tensor = torch.FloatTensor(targets).to(device)
    
    # Optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # División en batches
    num_samples = len(inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"🚀 Iniciando entrenamiento...")
    print(f"   Muestras: {num_samples}")
    print(f"   Épocas: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}\n")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Mezclar datos
        indices = np.random.permutation(num_samples)
        inputs_shuffled = inputs_tensor[indices]
        targets_shuffled = targets_tensor[indices]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_inputs = inputs_shuffled[start_idx:end_idx]
            batch_targets = targets_shuffled[start_idx:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            
            # Calcular pérdida
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch + 1}/{epochs}, Pérdida promedio: {avg_loss:.6f}")
    
    model.eval()
    print(f"\n✅ Entrenamiento completado!")
    return model


def main():
    print("=" * 60)
    print("🧠 Entrenamiento de Red Neuronal para Control de TurtleSim")
    print("=" * 60)
    
    # Generar datos de entrenamiento
    print("\n📊 Generando datos de entrenamiento...")
    inputs, targets = generate_training_data(num_samples=5000)
    print(f"✅ Generados {len(inputs)} ejemplos de entrenamiento")
    
    # Crear modelo
    model = NeuralNetworkController()
    
    # Entrenar modelo
    trained_model = train_model(
        model, 
        inputs, 
        targets, 
        epochs=100, 
        batch_size=32, 
        learning_rate=0.001
    )
    
    # Guardar modelo
    model_path = 'turtle_nn_model.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")
    
    # Evaluación rápida
    print("\n📈 Evaluación del modelo:")
    trained_model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor(inputs[:10])
        predictions = trained_model(test_inputs)
        print(f"   Ejemplo 1 - Entrada: {inputs[0]}")
        print(f"              Predicción: {predictions[0].detach().numpy()}")
        print(f"              Objetivo: {targets[0]}")
        print(f"              Error: {np.abs(predictions[0].detach().numpy() - targets[0])}")


if __name__ == '__main__':
    main()

