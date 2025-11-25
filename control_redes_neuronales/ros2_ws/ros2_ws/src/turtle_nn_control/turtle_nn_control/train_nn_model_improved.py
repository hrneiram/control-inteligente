#!/usr/bin/env python3
"""
Script de entrenamiento mejorado con hiperparámetros configurables
y generación de datos mejorada para mejor evitación de obstáculos
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import argparse
from turtle_nn_controller import NeuralNetworkController


def generate_training_data_improved(num_samples=10000, obstacle_focus=0.4):
    """
    Genera datos de entrenamiento mejorados con mejor balance de escenarios
    
    Args:
        num_samples: Número total de ejemplos
        obstacle_focus: Proporción de ejemplos con obstáculos cercanos (0.0-1.0)
    """
    inputs = []
    targets = []
    
    num_critical = int(num_samples * obstacle_focus)
    num_normal = num_samples - num_critical
    
    print(f"📊 Generando {num_critical} ejemplos críticos y {num_normal} normales...")
    
    for i in range(num_samples):
        # Determinar tipo de escenario
        if i < num_critical:
            # Escenarios críticos: obstáculo muy cercano o en trayectoria
            dist_to_obstacle = np.random.uniform(0.1, 1.5)
            dist_to_goal = np.random.uniform(0.5, 12.0)
        else:
            # Escenarios normales
            dist_to_obstacle = np.random.uniform(1.5, 15.0)
            dist_to_goal = np.random.uniform(0.5, 15.0)
        
        angle_to_goal = np.random.uniform(-math.pi, math.pi)
        angle_to_obstacle = np.random.uniform(-math.pi, math.pi)
        current_linear_vel = np.random.uniform(0.0, 2.0)
        current_angular_vel = np.random.uniform(-3.0, 3.0)
        
        # Calcular salidas deseadas con reglas mejoradas
        # Regla 1: Obstáculo MUY cercano (crítico)
        if dist_to_obstacle < 0.5:
            linear_vel = 0.1  # Muy lento o retroceder
            # Giro urgente lejos del obstáculo
            if abs(angle_to_obstacle) < math.pi / 2:
                angular_vel = 3.0 * np.sign(angle_to_obstacle) if angle_to_obstacle != 0 else 3.0
            else:
                # Si el obstáculo está detrás, girar hacia el objetivo
                angular_vel = 2.0 * np.sign(angle_to_goal)
        
        # Regla 2: Obstáculo cercano (alta urgencia)
        elif dist_to_obstacle < 1.5:
            linear_vel = 0.3
            # Verificar si la trayectoria hacia el objetivo pasa cerca del obstáculo
            angle_diff_to_goal = abs(angle_to_obstacle - angle_to_goal)
            angle_diff_to_goal = min(angle_diff_to_goal, 2*math.pi - angle_diff_to_goal)
            
            if angle_diff_to_goal < math.pi / 3:  # Trayectoria peligrosa
                # Priorizar evitación
                angular_vel = 2.8 * np.sign(angle_to_obstacle) if angle_to_obstacle != 0 else 2.8
                linear_vel = 0.2  # Aún más lento
            else:
                # Combinar evitación y navegación
                if abs(angle_to_obstacle) < math.pi / 2:
                    angular_vel = 2.2 * np.sign(angle_to_obstacle)
                else:
                    angular_vel = 1.5 * np.sign(angle_to_goal)
        
        # Regla 3: Obstáculo a distancia media
        elif dist_to_obstacle < 3.0:
            linear_vel = 0.7
            # Verificar si la trayectoria es peligrosa
            angle_diff_to_goal = abs(angle_to_obstacle - angle_to_goal)
            angle_diff_to_goal = min(angle_diff_to_goal, 2*math.pi - angle_diff_to_goal)
            
            if angle_diff_to_goal < math.pi / 2:
                # Preparar para evitar
                angular_vel = 1.8 * np.sign(angle_to_obstacle) if angle_to_obstacle != 0 else 1.8
            else:
                # Navegar normalmente pero con precaución
                angular_vel = 1.2 * np.sign(angle_to_goal)
        
        # Regla 4: Sin obstáculos cercanos, navegar normalmente
        else:
            # Velocidad basada en distancia al objetivo
            if dist_to_goal < 1.0:
                linear_vel = 0.6
            elif dist_to_goal < 3.0:
                linear_vel = 1.3
            else:
                linear_vel = 1.6
            
            # Velocidad angular basada en error de ángulo
            angle_error = angle_to_goal
            angle_abs = abs(angle_error)
            
            if angle_abs > 1.5:
                angular_vel = 2.0 * np.sign(angle_error)
            elif angle_abs > 1.0:
                angular_vel = 1.5 * np.sign(angle_error)
            elif angle_abs > 0.5:
                angular_vel = 1.0 * np.sign(angle_error)
            else:
                angular_vel = 0.5 * np.sign(angle_error)
        
        # Normalizar entradas
        angle_to_obstacle_rel = angle_to_obstacle - angle_to_goal
        angle_to_obstacle_rel = math.atan2(
            math.sin(angle_to_obstacle_rel),
            math.cos(angle_to_obstacle_rel)
        )
        
        inputs.append([
            min(dist_to_goal / 15.0, 1.0),
            angle_to_goal / math.pi,
            min(dist_to_obstacle / 15.0, 1.0),
            angle_to_obstacle_rel / math.pi if dist_to_obstacle < 15.0 else 0.0,
            current_linear_vel / 2.0,
            current_angular_vel / 3.0
        ])
        
        # Targets NO normalizados
        targets.append([
            linear_vel,
            angular_vel
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
    
    # Learning rate scheduler para fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # División en batches
    num_samples = len(inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"🚀 Iniciando entrenamiento...")
    print(f"   Muestras: {num_samples}")
    print(f"   Épocas: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}\n")
    
    best_loss = float('inf')
    
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
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Época {epoch + 1}/{epochs}, Pérdida: {avg_loss:.6f}, "
                  f"Mejor: {best_loss:.6f}, LR: {current_lr:.6f}")
    
    model.eval()
    print(f"\n✅ Entrenamiento completado!")
    print(f"📊 Mejor pérdida alcanzada: {best_loss:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo neuronal con parámetros mejorados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Entrenamiento básico mejorado
  python3 train_nn_model_improved.py
  
  # Más énfasis en evitación de obstáculos
  python3 train_nn_model_improved.py --obstacle_focus 0.5 --samples 15000
  
  # Fine-tuning con más épocas
  python3 train_nn_model_improved.py --epochs 200 --lr 0.0005
        """
    )
    
    parser.add_argument('--epochs', type=int, default=150,
                       help='Número de épocas (default: 150)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño del batch (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate inicial (default: 0.001)')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Número de muestras de entrenamiento (default: 10000)')
    parser.add_argument('--obstacle_focus', type=float, default=0.4,
                       help='Proporción de ejemplos con obstáculos cercanos (0.0-1.0, default: 0.4)')
    parser.add_argument('--hidden1', type=int, default=64,
                       help='Tamaño de la primera capa oculta (default: 64)')
    parser.add_argument('--hidden2', type=int, default=32,
                       help='Tamaño de la segunda capa oculta (default: 32)')
    parser.add_argument('--output', type=str, default='turtle_nn_model.pth',
                       help='Nombre del archivo de salida (default: turtle_nn_model.pth)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧠 Entrenamiento Mejorado de Red Neuronal para Control de TurtleSim")
    print("=" * 70)
    print(f"\n📋 Parámetros de Entrenamiento:")
    print(f"   Épocas: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Muestras: {args.samples}")
    print(f"   Foco en obstáculos: {args.obstacle_focus*100:.0f}%")
    print(f"   Arquitectura: 6 → {args.hidden1} → {args.hidden2} → 2")
    print()
    
    # Generar datos mejorados
    print("📊 Generando datos de entrenamiento mejorados...")
    inputs, targets = generate_training_data_improved(
        num_samples=args.samples,
        obstacle_focus=args.obstacle_focus
    )
    print(f"✅ Generados {len(inputs)} ejemplos de entrenamiento")
    
    # Crear modelo
    model = NeuralNetworkController(
        input_size=6,
        hidden_size1=args.hidden1,
        hidden_size2=args.hidden2,
        output_size=2
    )
    
    # Entrenar modelo
    trained_model = train_model(
        model,
        inputs,
        targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Guardar modelo
    torch.save(trained_model.state_dict(), args.output)
    print(f"\n💾 Modelo guardado en: {args.output}")
    
    # Evaluación rápida
    print("\n📈 Evaluación del modelo:")
    trained_model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor(inputs[:10])
        predictions = trained_model(test_inputs)
        print(f"   Ejemplo 1:")
        print(f"      Entrada: {inputs[0]}")
        print(f"      Predicción: {predictions[0].detach().numpy()}")
        print(f"      Objetivo: {targets[0]}")
        error = np.abs(predictions[0].detach().numpy() - targets[0])
        print(f"      Error: {error} (Total: {np.sum(error):.4f})")
    
    print("\n✅ ¡Entrenamiento completado exitosamente!")
    print("\n📝 Próximos pasos:")
    print("   1. Recompilar: cd /ros2_ws && colcon build --packages-select turtle_nn_control")
    print("   2. Copiar modelo: cp turtle_nn_model.pth install/turtle_nn_control/lib/python3.10/site-packages/turtle_nn_control/")
    print("   3. Ejecutar: source install/setup.bash && ros2 run turtle_nn_control nn_controller")


if __name__ == '__main__':
    main()

