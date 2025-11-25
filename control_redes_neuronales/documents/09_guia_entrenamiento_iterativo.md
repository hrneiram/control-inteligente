# Guía Extensa de Entrenamiento Iterativo del Modelo Neuronal

Esta guía detalla el proceso completo de entrenamiento, compilación e iteración para mejorar el rendimiento del controlador neuronal de TurtleSim.

---

## Tabla de Contenidos

1. [Flujo de Trabajo Completo](#flujo-de-trabajo-completo)
2. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
3. [Compilación y Despliegue](#compilación-y-despliegue)
4. [Evaluación y Análisis](#evaluación-y-análisis)
5. [Iteración y Mejora](#iteración-y-mejora)
6. [Ajuste de Hiperparámetros](#ajuste-de-hiperparámetros)
7. [Mejora de Datos de Entrenamiento](#mejora-de-datos-de-entrenamiento)
8. [Debugging y Solución de Problemas](#debugging-y-solución-de-problemas)
9. [Checklist de Mejora Continua](#checklist-de-mejora-continua)

---

## Flujo de Trabajo Completo

### Diagrama del Proceso

```
1. Modificar parámetros/datos
   ↓
2. Entrenar modelo
   ↓
3. Compilar paquete
   ↓
4. Copiar modelo
   ↓
5. Ejecutar y observar
   ↓
6. Analizar comportamiento
   ↓
7. Identificar problemas
   ↓
8. Volver al paso 1
```

---

## Entrenamiento del Modelo

### Paso 1: Preparar el Entorno

```bash
# Dentro del contenedor Docker
cd /ros2_ws/src/turtle_nn_control/turtle_nn_control
```

### Paso 2: Entrenar el Modelo Básico

```bash
python3 train_nn_model.py
```

**Parámetros por defecto:**
- Muestras: 5000
- Épocas: 100
- Batch size: 32
- Learning rate: 0.001

### Paso 3: Entrenar con Parámetros Personalizados

Puedes modificar `train_nn_model.py` para cambiar los parámetros:

```python
# En la función main(), cambiar:
trained_model = train_model(
    model, 
    inputs, 
    targets, 
    epochs=200,        # Más épocas para mejor convergencia
    batch_size=64,    # Batch más grande para estabilidad
    learning_rate=0.0005  # Learning rate más bajo para fine-tuning
)
```

### Paso 4: Verificar el Entrenamiento

Observa la salida del entrenamiento:

```
Época 10/100, Pérdida promedio: 0.260246
Época 20/100, Pérdida promedio: 0.146833
...
Época 100/100, Pérdida promedio: 0.045450
```

**Indicadores de buen entrenamiento:**
- ✅ Pérdida disminuye consistentemente
- ✅ Pérdida final < 0.1 (idealmente < 0.05)
- ✅ No hay sobreajuste (pérdida no aumenta al final)

**Señales de problemas:**
- ❌ Pérdida no disminuye: learning rate muy bajo o arquitectura insuficiente
- ❌ Pérdida aumenta: learning rate muy alto
- ❌ Pérdida oscila: batch size muy pequeño

---

## Compilación y Despliegue

### Método 1: Script Automatizado (Recomendado)

```bash
cd /ros2_ws
./src/turtle_nn_control/rebuild_and_setup.sh
source install/setup.bash
```

### Método 2: Manual (Paso a Paso)

```bash
# 1. Ir al workspace
cd /ros2_ws

# 2. Recompilar el paquete
colcon build --packages-select turtle_nn_control

# 3. Verificar que el modelo existe
ls -la src/turtle_nn_control/turtle_nn_control/turtle_nn_model.pth

# 4. Copiar el modelo al directorio de instalación
mkdir -p install/turtle_nn_control/lib/python3.10/site-packages/turtle_nn_control
cp src/turtle_nn_control/turtle_nn_control/turtle_nn_model.pth \
   install/turtle_nn_control/lib/python3.10/site-packages/turtle_nn_control/

# 5. Recargar el entorno
source install/setup.bash

# 6. Verificar que el modelo se puede cargar
python3 -c "import torch; print('PyTorch OK'); m = torch.load('install/turtle_nn_control/lib/python3.10/site-packages/turtle_nn_control/turtle_nn_model.pth'); print('Modelo OK')"
```

### Verificación Rápida

```bash
# Verificar que el paquete está instalado
ros2 pkg list | grep turtle_nn_control

# Verificar que el ejecutable existe
ros2 run turtle_nn_control nn_controller --help
```

---

## Evaluación y Análisis

### Paso 1: Ejecutar el Simulador

**Terminal 1:**
```bash
ros2 run turtlesim turtlesim_node
```

### Paso 2: Ejecutar el Controlador

**Terminal 2:**
```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 run turtle_nn_control nn_controller
```

### Paso 3: Observar el Comportamiento

**Métricas a observar:**

1. **Navegación hacia objetivos:**
   - ✅ ¿Llega a los objetivos de forma directa?
   - ✅ ¿Tarda mucho tiempo?
   - ✅ ¿Hace movimientos innecesarios?

2. **Evitación de obstáculos:**
   - ✅ ¿Se desvía a tiempo?
   - ✅ ¿Mantiene distancia segura?
   - ❌ ¿Choca con obstáculos?
   - ❌ ¿Se queda atascado cerca de obstáculos?

3. **Suavidad del movimiento:**
   - ✅ ¿Movimientos fluidos?
   - ❌ ¿Movimientos bruscos o erráticos?
   - ❌ ¿Oscilaciones?

### Paso 4: Registrar Observaciones

Crea un archivo de log para cada iteración:

```bash
# Crear directorio de logs
mkdir -p /ros2_ws/training_logs

# Registrar observaciones
cat > /ros2_ws/training_logs/iteracion_01.md << EOF
# Iteración 01 - [Fecha]

## Parámetros de Entrenamiento
- Épocas: 100
- Batch size: 32
- Learning rate: 0.001
- Muestras: 5000

## Observaciones
- ✅ Llega a objetivos correctamente
- ❌ A veces no se desvía de obstáculos
- ❌ Movimientos un poco bruscos cerca de obstáculos

## Problemas Identificados
1. El modelo no prioriza suficiente la evitación cuando el obstáculo está en la trayectoria
2. La velocidad angular es demasiado alta en algunas situaciones

## Próximos Pasos
1. Aumentar peso de evitación en datos de entrenamiento
2. Reducir velocidad angular máxima en reglas heurísticas
EOF
```

---

## Iteración y Mejora

### Ciclo de Iteración Recomendado

```
┌─────────────────────────────────┐
│ 1. Identificar problema        │
└────────────┬──────────────────┘
             ↓
┌─────────────────────────────────┐
│ 2. Modificar datos/parámetros  │
└────────────┬──────────────────┘
             ↓
┌─────────────────────────────────┐
│ 3. Entrenar nuevo modelo       │
└────────────┬──────────────────┘
             ↓
┌─────────────────────────────────┐
│ 4. Compilar y desplegar        │
└────────────┬──────────────────┘
             ↓
┌─────────────────────────────────┐
│ 5. Evaluar comportamiento       │
└────────────┬──────────────────┘
             ↓
┌─────────────────────────────────┐
│ 6. Comparar con iteración prev │
└────────────┬──────────────────┘
             ↓
         ¿Mejoró?
         /      \
       Sí        No
       ↓         ↓
    Guardar   Analizar
    modelo    más profundo
```

### Ejemplo de Iteración: Mejorar Evitación de Obstáculos

**Problema identificado:** El modelo no se desvía suficientemente de obstáculos.

**Solución 1: Aumentar urgencia de evitación en datos**

Modificar `train_nn_model.py` en la función `generate_training_data()`:

```python
# ANTES (línea ~40):
if dist_to_obstacle < 1.0:
    linear_vel = 0.2
    angular_vel = 2.5 * np.sign(angle_to_obstacle)

# DESPUÉS (más agresivo):
if dist_to_obstacle < 1.5:  # Aumentar rango de urgencia
    linear_vel = 0.15  # Más lento
    angular_vel = 2.8 * np.sign(angle_to_obstacle)  # Giro más fuerte
```

**Solución 2: Generar más ejemplos de evitación**

```python
# Aumentar proporción de ejemplos con obstáculos cercanos
for _ in range(num_samples):
    # 30% de ejemplos con obstáculos muy cercanos
    if np.random.random() < 0.3:
        dist_to_obstacle = np.random.uniform(0.1, 1.5)  # Más ejemplos cercanos
    else:
        dist_to_obstacle = np.random.uniform(0.1, 15.0)
```

**Solución 3: Aumentar muestras de entrenamiento**

```python
# En main():
inputs, targets = generate_training_data(num_samples=10000)  # Más datos
```

---

## Ajuste de Hiperparámetros

### Tabla de Hiperparámetros y Efectos

| Hiperparámetro | Valor Actual | Aumentar | Disminuir |
|---------------|--------------|----------|-----------|
| **Épocas** | 100 | Más entrenamiento, riesgo de sobreajuste | Menos entrenamiento, puede no converger |
| **Batch Size** | 32 | Más estable, más memoria | Menos estable, más rápido |
| **Learning Rate** | 0.001 | Convergencia más rápida, puede oscilar | Convergencia más lenta, más estable |
| **Hidden Size 1** | 64 | Más capacidad, más parámetros | Menos capacidad, más rápido |
| **Hidden Size 2** | 32 | Más capacidad, más parámetros | Menos capacidad, más rápido |
| **Dropout** | 0.1 | Más regularización | Menos regularización |

### Guía de Ajuste por Problema

#### Problema: Modelo no aprende (pérdida no disminuye)

```python
# Soluciones:
1. Aumentar learning rate: 0.001 → 0.002
2. Aumentar tamaño de capas: 64 → 128, 32 → 64
3. Reducir dropout: 0.1 → 0.05
4. Aumentar épocas: 100 → 200
```

#### Problema: Modelo sobreajusta (pérdida aumenta al final)

```python
# Soluciones:
1. Aumentar dropout: 0.1 → 0.2
2. Reducir learning rate: 0.001 → 0.0005
3. Aumentar batch size: 32 → 64
4. Reducir tamaño de capas: 64 → 48, 32 → 24
```

#### Problema: Convergencia muy lenta

```python
# Soluciones:
1. Aumentar learning rate: 0.001 → 0.002
2. Reducir épocas pero aumentar batch: 100 → 150, 32 → 64
3. Usar scheduler de learning rate
```

### Script de Entrenamiento con Hiperparámetros Configurables

Crea un archivo `train_with_params.py`:

```python
#!/usr/bin/env python3
"""
Script de entrenamiento con hiperparámetros configurables
Uso: python3 train_with_params.py --epochs 200 --lr 0.0005
"""

import argparse
from train_nn_model import generate_training_data, train_model, NeuralNetworkController
import torch

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo con parámetros personalizados')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=5000, help='Número de muestras')
    parser.add_argument('--hidden1', type=int, default=64, help='Tamaño capa oculta 1')
    parser.add_argument('--hidden2', type=int, default=32, help='Tamaño capa oculta 2')
    parser.add_argument('--output', type=str, default='turtle_nn_model.pth', help='Nombre del archivo de salida')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 Entrenamiento con Parámetros Personalizados")
    print("=" * 60)
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Muestras: {args.samples}")
    print(f"Arquitectura: 6 → {args.hidden1} → {args.hidden2} → 2")
    print()
    
    # Generar datos
    print("📊 Generando datos...")
    inputs, targets = generate_training_data(num_samples=args.samples)
    print(f"✅ {len(inputs)} ejemplos generados")
    
    # Crear modelo con arquitectura personalizada
    model = NeuralNetworkController(
        input_size=6,
        hidden_size1=args.hidden1,
        hidden_size2=args.hidden2,
        output_size=2
    )
    
    # Entrenar
    trained_model = train_model(
        model,
        inputs,
        targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Guardar
    torch.save(trained_model.state_dict(), args.output)
    print(f"\n💾 Modelo guardado en: {args.output}")

if __name__ == '__main__':
    main()
```

**Uso:**
```bash
# Entrenar con parámetros personalizados
python3 train_with_params.py --epochs 200 --lr 0.0005 --samples 10000

# Entrenar con arquitectura más grande
python3 train_with_params.py --hidden1 128 --hidden2 64 --epochs 150
```

---

## Mejora de Datos de Entrenamiento

### Estrategias para Mejorar los Datos

#### 1. Aumentar Diversidad de Escenarios

```python
def generate_training_data_improved(num_samples=5000):
    inputs = []
    targets = []
    
    # 40% escenarios con obstáculos cercanos (críticos)
    # 30% escenarios con obstáculos medios
    # 30% escenarios sin obstáculos cercanos
    
    for i in range(num_samples):
        if i < num_samples * 0.4:
            # Obstáculos muy cercanos
            dist_to_obstacle = np.random.uniform(0.1, 1.5)
        elif i < num_samples * 0.7:
            # Obstáculos medios
            dist_to_obstacle = np.random.uniform(1.5, 3.0)
        else:
            # Sin obstáculos cercanos
            dist_to_obstacle = np.random.uniform(3.0, 15.0)
        
        # ... resto del código
```

#### 2. Añadir Penalización por Trayectorias Peligrosas

```python
# En las reglas heurísticas, detectar si la trayectoria pasa cerca del obstáculo
angle_to_goal = np.random.uniform(-math.pi, math.pi)
angle_to_obstacle = np.random.uniform(-math.pi, math.pi)

# Si la trayectoria hacia el objetivo pasa cerca del obstáculo
angle_diff = abs(angle_to_goal - angle_to_obstacle)
if angle_diff < math.pi / 3 and dist_to_obstacle < 2.5:
    # Aumentar urgencia de evitación
    linear_vel *= 0.7  # Reducir más la velocidad
    angular_vel = 2.2 * np.sign(angle_to_obstacle)  # Giro más agresivo
```

#### 3. Añadir Ejemplos de Recuperación

```python
# Ejemplos donde el robot está muy cerca y necesita escapar
if dist_to_obstacle < 0.3:
    # Escenario crítico: retroceder y girar
    linear_vel = -0.3  # Retroceder
    angular_vel = 3.0 * np.sign(angle_to_obstacle)  # Giro máximo
```

#### 4. Balancear Datos por Distancia al Objetivo

```python
# Asegurar ejemplos en todas las distancias
if i % 3 == 0:
    dist_to_goal = np.random.uniform(0.1, 1.0)  # Cerca
elif i % 3 == 1:
    dist_to_goal = np.random.uniform(1.0, 5.0)  # Media
else:
    dist_to_goal = np.random.uniform(5.0, 15.0)  # Lejos
```

### Función Mejorada de Generación de Datos

Crea `train_nn_model_improved.py` con estas mejoras:

```python
def generate_training_data_improved(num_samples=10000):
    """
    Versión mejorada con más diversidad y escenarios críticos
    """
    inputs = []
    targets = []
    
    for i in range(num_samples):
        # Balancear escenarios
        scenario_type = np.random.choice(['critical', 'medium', 'normal'], 
                                        p=[0.3, 0.3, 0.4])
        
        if scenario_type == 'critical':
            # Escenarios críticos: obstáculo muy cercano
            dist_to_obstacle = np.random.uniform(0.1, 1.0)
            dist_to_goal = np.random.uniform(0.5, 10.0)
        elif scenario_type == 'medium':
            # Escenarios medios: obstáculo a distancia media
            dist_to_obstacle = np.random.uniform(1.0, 3.0)
            dist_to_goal = np.random.uniform(1.0, 12.0)
        else:
            # Escenarios normales: sin obstáculos cercanos
            dist_to_obstacle = np.random.uniform(3.0, 15.0)
            dist_to_goal = np.random.uniform(0.5, 15.0)
        
        # ... resto de la generación con reglas mejoradas
```

---

## Debugging y Solución de Problemas

### Problema: Modelo no se desvía de obstáculos

**Diagnóstico:**
```python
# Añadir logging en control_loop para ver qué está pasando
def control_loop(self):
    # ... código existente ...
    
    # DEBUG: Log cuando hay obstáculo cercano
    if dist_to_obstacle < 2.0:
        self.get_logger().warn(
            f'🚨 OBSTÁCULO CERCANO! Dist: {dist_to_obstacle:.2f}, '
            f'Urgencia esperada: alta, '
            f'Velocidades: lin={linear_vel:.2f}, ang={angular_vel:.2f}'
        )
```

**Soluciones:**

1. **Aumentar peso de evitación en entrenamiento:**
   ```python
   # En generate_training_data, hacer las reglas más estrictas
   if dist_to_obstacle < 2.0:  # Aumentar rango
       linear_vel = 0.4  # Más conservador
       angular_vel = 2.5 * np.sign(angle_to_obstacle)
   ```

2. **Aumentar sensibilidad en el controlador:**
   ```python
   # En control_loop, ajustar umbral
   if dist_to_obstacle < 2.5:  # Detectar antes (era 2.0)
       # Forzar más reducción de velocidad
       linear_vel *= 0.7
   ```

### Problema: Movimientos bruscos

**Solución: Suavizar salidas**

```python
# En control_loop, añadir filtro de suavizado
if not hasattr(self, 'prev_linear_vel'):
    self.prev_linear_vel = 0.0
    self.prev_angular_vel = 0.0

# Suavizar con promedio móvil (alpha = 0.7)
alpha = 0.7
linear_vel = alpha * linear_vel + (1 - alpha) * self.prev_linear_vel
angular_vel = alpha * angular_vel + (1 - alpha) * self.prev_angular_vel

self.prev_linear_vel = linear_vel
self.prev_angular_vel = angular_vel
```

### Problema: No llega a objetivos

**Solución: Aumentar prioridad de navegación**

```python
# En generate_training_data, cuando no hay obstáculos:
if dist_to_obstacle > 3.0:
    # Aumentar velocidad hacia objetivo
    if dist_to_goal < 1.0:
        linear_vel = 0.6  # Más rápido cerca del objetivo
    elif dist_to_goal < 3.0:
        linear_vel = 1.4  # Más rápido en distancia media
    else:
        linear_vel = 1.8  # Más rápido cuando está lejos
```

---

## Checklist de Mejora Continua

### Antes de Cada Iteración

- [ ] Identificar problema específico a resolver
- [ ] Revisar logs de iteración anterior
- [ ] Decidir qué cambiar (datos, hiperparámetros, o ambos)
- [ ] Hacer backup del modelo anterior: `cp turtle_nn_model.pth turtle_nn_model_backup.pth`

### Durante el Entrenamiento

- [ ] Verificar que la pérdida disminuye
- [ ] Observar que no hay sobreajuste
- [ ] Anotar pérdida final para comparación

### Después del Entrenamiento

- [ ] Recompilar el paquete
- [ ] Copiar el modelo al directorio de instalación
- [ ] Ejecutar y observar comportamiento
- [ ] Comparar con iteración anterior
- [ ] Documentar resultados en log de iteración

### Métricas de Éxito

- [ ] ✅ Llega a objetivos > 90% de las veces
- [ ] ✅ Evita obstáculos > 95% de las veces
- [ ] ✅ No se queda atascado
- [ ] ✅ Movimientos suaves y naturales
- [ ] ✅ Tiempo promedio de llegada razonable

---

## Ejemplo Completo de Iteración

### Iteración 1: Modelo Base

```bash
# Entrenar
python3 train_nn_model.py
# Resultado: Pérdida final: 0.045

# Observación: A veces no se desvía de obstáculos
```

### Iteración 2: Mejorar Evitación

```bash
# Modificar train_nn_model.py:
# - Aumentar rango de urgencia: 1.0 → 1.5
# - Aumentar giro: 2.5 → 2.8
# - 30% más ejemplos con obstáculos cercanos

python3 train_nn_model.py
# Resultado: Pérdida final: 0.038

# Observación: Mejor evitación, pero movimientos más bruscos
```

### Iteración 3: Suavizar Movimientos

```bash
# Añadir suavizado en control_loop
# Reducir learning rate: 0.001 → 0.0008

python3 train_nn_model.py
# Resultado: Pérdida final: 0.042

# Observación: Movimientos más suaves, evitación mantenida
```

### Iteración 4: Fine-tuning

```bash
# Aumentar muestras: 5000 → 8000
# Aumentar épocas: 100 → 150
# Learning rate scheduler

python3 train_nn_model.py
# Resultado: Pérdida final: 0.031

# Observación: Comportamiento óptimo
```

---

## Recursos Adicionales

### Scripts Útiles

1. **`compare_models.py`**: Compara dos modelos lado a lado
2. **`visualize_training.py`**: Genera gráficas de pérdida
3. **`test_scenarios.py`**: Prueba el modelo en escenarios específicos

### Comandos Rápidos

```bash
# Entrenar y desplegar en un comando
python3 train_nn_model.py && \
cd /ros2_ws && \
colcon build --packages-select turtle_nn_control && \
cp src/turtle_nn_control/turtle_nn_control/turtle_nn_model.pth \
   install/turtle_nn_control/lib/python3.10/site-packages/turtle_nn_control/ && \
source install/setup.bash && \
echo "✅ Listo para ejecutar: ros2 run turtle_nn_control nn_controller"
```

---

## Conclusión

El proceso de mejora iterativa requiere:

1. **Paciencia**: Cada iteración puede tomar tiempo
2. **Observación cuidadosa**: Identificar problemas específicos
3. **Cambios incrementales**: No cambiar todo a la vez
4. **Documentación**: Registrar cada iteración
5. **Comparación**: Comparar con iteraciones anteriores

Con esta guía, deberías poder mejorar sistemáticamente el rendimiento del modelo hasta alcanzar un comportamiento óptimo.

