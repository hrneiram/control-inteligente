#!/usr/bin/env python3
"""
Control basado en Redes Neuronales para TurtleSim
Usa una red neuronal feedforward para navegación y evitación de obstáculos
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import Spawn, Kill, TeleportAbsolute, SetPen
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Obstacle:
    """Representa un obstáculo circular"""
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
    
    def distance_to(self, x, y):
        """Distancia desde un punto al borde del obstáculo"""
        dist_to_center = math.sqrt((x - self.x)**2 + (y - self.y)**2)
        return max(0, dist_to_center - self.radius)


class NeuralNetworkController(nn.Module):
    """
    Red neuronal feedforward para control de navegación
    Entradas: [dist_to_goal, angle_to_goal, dist_to_obstacle, angle_to_obstacle, 
               current_linear_vel, current_angular_vel]
    Salidas: [linear_velocity, angular_velocity]
    """
    def __init__(self, input_size=6, hidden_size1=64, hidden_size2=32, output_size=2):
        super(NeuralNetworkController, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Normalizar salidas: linear_vel [0, 2.0], angular_vel [-3.0, 3.0]
        # x tiene forma [batch_size, 2], donde x[:, 0] es linear y x[:, 1] es angular
        linear_vel = torch.sigmoid(x[:, 0]) * 2.0  # [0, 2.0]
        angular_vel = torch.tanh(x[:, 1]) * 3.0    # [-3.0, 3.0]
        # Apilar en la dimensión 1 para obtener [batch_size, 2]
        return torch.stack([linear_vel, angular_vel], dim=1)


class TurtleNNController(Node):
    def __init__(self):
        super().__init__('turtle_nn_controller')
        
        # Publisher para comandos de velocidad
        self.velocity_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Subscriber para la pose actual
        self.pose_subscriber = self.create_subscription(
            Pose, '/turtle1/pose', self.pose_callback, 10)
        
        # Variables de estado
        self.current_pose = Pose()
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        
        # Posición inicial segura
        self.start_x = 2.0
        self.start_y = 2.0
        
        # Objetivo inicial
        self.goal_x = 9.0
        self.goal_y = 9.0
        self.obstacles_drawn = False
        
        # Definir obstáculos (x, y, radio)
        self.obstacles = [
            Obstacle(5.5, 5.5, 0.4),   # Centro
            Obstacle(3.5, 8.0, 0.4),   # Superior izquierda
            Obstacle(8.0, 3.5, 0.4),   # Inferior derecha
        ]
        
        # Inicializar red neuronal
        self.device = torch.device('cpu')
        self.model = NeuralNetworkController().to(self.device)
        self.model.eval()  # Modo evaluación
        
        # Cargar modelo pre-entrenado si existe, sino usar pesos aleatorios
        # Buscar el modelo en varios lugares posibles
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'turtle_nn_model.pth'),  # En el directorio del módulo
            os.path.join('/ros2_ws/src/turtle_nn_control/turtle_nn_control', 'turtle_nn_model.pth'),  # En el workspace fuente
            'turtle_nn_model.pth'  # En el directorio actual
        ]
        
        model_loaded = False
        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.get_logger().info(f'✅ Modelo pre-entrenado cargado desde: {model_path}')
                    model_loaded = True
                    break
                except Exception as e:
                    self.get_logger().warn(f'⚠️  No se pudo cargar el modelo desde {model_path}: {e}')
        
        if not model_loaded:
            self.get_logger().info('ℹ️  Usando modelo con pesos aleatorios (puedes entrenar el modelo)')
        
        self.get_logger().info('🧠 Control Neuronal iniciado!')
        self.get_logger().info('⏳ Moviendo tortuga a posición inicial segura...')
        
        # Mover tortuga a posición inicial
        self.move_turtle_to_start()
        
        self.get_logger().info('⏳ Dibujando obstáculos y objetivo...')
        
        # Timer para dibujar obstáculos (se ejecuta una vez)
        self.draw_timer = self.create_timer(1.5, self.draw_obstacles_once)
        
        # Timer para control (se inicia después)
        self.control_timer = None
        
        # Contador para logging
        self.log_counter = 0
    
    def move_turtle_to_start(self):
        """Mueve turtle1 a la posición inicial segura"""
        client = self.create_client(TeleportAbsolute, '/turtle1/teleport_absolute')
        while not client.wait_for_service(timeout_sec=1.0):
            pass
        
        request = TeleportAbsolute.Request()
        request.x = self.start_x
        request.y = self.start_y
        request.theta = 0.0
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.get_logger().info(f'✅ Tortuga movida a ({self.start_x}, {self.start_y})')
    
    def draw_obstacles_once(self):
        """Dibuja los obstáculos una sola vez"""
        if self.obstacles_drawn:
            return
        
        # Dibujar obstáculos
        self.draw_obstacles()
        
        # Dibujar objetivo inicial
        self.draw_goal()
        
        self.obstacles_drawn = True
        self.draw_timer.cancel()
        
        # Iniciar control después de dibujar
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('🚧 Obstáculos y objetivo dibujados! Iniciando navegación neuronal...')
        self.get_logger().info(f'📍 Objetivo inicial: ({self.goal_x}, {self.goal_y})')
    
    def draw_obstacles(self):
        """Dibuja círculos rojos para representar obstáculos"""
        for i, obs in enumerate(self.obstacles):
            spawn_name = f'drawer_{i}'
            self.spawn_turtle(spawn_name, obs.x + obs.radius, obs.y)
            time.sleep(0.3)
            self.set_pen(spawn_name, 255, 0, 0, 3, 0)
            time.sleep(0.1)
            self.draw_circle(spawn_name, obs.radius)
            time.sleep(0.3)
            self.kill_turtle(spawn_name)
            time.sleep(0.1)
    
    def draw_goal(self):
        """Dibuja un círculo verde pequeño en la posición del objetivo"""
        spawn_name = 'goal_drawer'
        goal_radius = 0.3
        
        self.spawn_turtle(spawn_name, self.goal_x + goal_radius, self.goal_y)
        time.sleep(0.2)
        self.set_pen(spawn_name, 0, 255, 0, 2, 0)
        time.sleep(0.1)
        self.draw_circle(spawn_name, goal_radius)
        time.sleep(0.2)
        self.kill_turtle(spawn_name)
        time.sleep(0.1)
    
    def spawn_turtle(self, name, x, y):
        """Crea una tortuga temporal"""
        client = self.create_client(Spawn, '/spawn')
        while not client.wait_for_service(timeout_sec=1.0):
            pass
        
        request = Spawn.Request()
        request.x = x
        request.y = y
        request.theta = 0.0
        request.name = name
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
    
    def kill_turtle(self, name):
        """Elimina una tortuga"""
        client = self.create_client(Kill, '/kill')
        while not client.wait_for_service(timeout_sec=1.0):
            pass
        
        request = Kill.Request()
        request.name = name
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
    
    def set_pen(self, turtle_name, r, g, b, width, off):
        """Configura el lápiz de una tortuga"""
        client = self.create_client(SetPen, f'/{turtle_name}/set_pen')
        while not client.wait_for_service(timeout_sec=1.0):
            pass
        
        request = SetPen.Request()
        request.r = r
        request.g = g
        request.b = b
        request.width = width
        request.off = off
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
    
    def draw_circle(self, turtle_name, radius):
        """Dibuja un círculo moviendo la tortuga"""
        vel_pub = self.create_publisher(Twist, f'/{turtle_name}/cmd_vel', 10)
        time.sleep(0.1)
        
        linear_speed = 1.0
        angular_speed = linear_speed / radius
        circle_time = 2 * math.pi * radius / linear_speed
        
        cmd = Twist()
        cmd.linear.x = linear_speed
        cmd.angular.z = angular_speed
        
        start_time = time.time()
        while (time.time() - start_time) < circle_time:
            vel_pub.publish(cmd)
            time.sleep(0.02)
        
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        vel_pub.publish(cmd)
    
    def pose_callback(self, msg):
        """Actualiza la pose actual de la tortuga"""
        self.current_pose = msg
    
    def find_closest_obstacle(self, x, y):
        """Encuentra el obstáculo más cercano"""
        min_dist = float('inf')
        closest_obs = None
        
        for obs in self.obstacles:
            dist = obs.distance_to(x, y)
            if dist < min_dist:
                min_dist = dist
                closest_obs = obs
        
        return closest_obs, min_dist
    
    def normalize_inputs(self, dist_to_goal, angle_to_goal, dist_to_obstacle, 
                        angle_to_obstacle, linear_vel, angular_vel):
        """
        Normaliza las entradas para la red neuronal
        - dist_to_goal: [0, 15] -> [0, 1]
        - angle_to_goal: [-π, π] -> [-1, 1]
        - dist_to_obstacle: [0, 15] -> [0, 1]
        - angle_to_obstacle: [-π, π] -> [-1, 1]
        - linear_vel: [0, 2] -> [0, 1]
        - angular_vel: [-3, 3] -> [-1, 1]
        """
        return np.array([
            min(dist_to_goal / 15.0, 1.0),
            angle_to_goal / math.pi,
            min(dist_to_obstacle / 15.0, 1.0),
            angle_to_obstacle / math.pi if dist_to_obstacle < 15.0 else 0.0,
            linear_vel / 2.0,
            angular_vel / 3.0
        ], dtype=np.float32)
    
    def control_loop(self):
        """Bucle principal de control usando red neuronal"""
        if not self.obstacles_drawn:
            return
        
        # Calcular distancia y ángulo al objetivo
        dx = self.goal_x - self.current_pose.x
        dy = self.goal_y - self.current_pose.y
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        angle_to_goal = math.atan2(dy, dx)
        angle_error = angle_to_goal - self.current_pose.theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        # Encontrar obstáculo más cercano
        closest_obs, dist_to_obstacle = self.find_closest_obstacle(
            self.current_pose.x, self.current_pose.y)
        
        if closest_obs:
            angle_to_obstacle = math.atan2(
                closest_obs.y - self.current_pose.y,
                closest_obs.x - self.current_pose.x)
            angle_to_obstacle_rel = angle_to_obstacle - self.current_pose.theta
            angle_to_obstacle_rel = math.atan2(
                math.sin(angle_to_obstacle_rel), 
                math.cos(angle_to_obstacle_rel))
        else:
            dist_to_obstacle = 15.0  # Muy lejos
            angle_to_obstacle_rel = 0.0
        
        # Normalizar entradas
        inputs = self.normalize_inputs(
            distance_to_goal,
            angle_error,
            dist_to_obstacle,
            angle_to_obstacle_rel,
            self.current_linear_vel,
            self.current_angular_vel
        )
        
        # Pasar por la red neuronal
        with torch.no_grad():
            input_tensor = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            linear_vel = output[0, 0].item()
            angular_vel = output[0, 1].item()
        
        # Actualizar velocidades actuales
        self.current_linear_vel = linear_vel
        self.current_angular_vel = angular_vel
        
        # Verificar llegada al objetivo
        if distance_to_goal < 0.5:
            self.get_logger().info('🎉 ¡Objetivo alcanzado!')
            
            # Generar nuevo objetivo en posición segura
            safe_goal_found = False
            attempts = 0
            while not safe_goal_found and attempts < 20:
                new_x = np.random.uniform(2.0, 9.0)
                new_y = np.random.uniform(2.0, 9.0)
                
                min_dist_to_obstacles = min([obs.distance_to(new_x, new_y) 
                                            for obs in self.obstacles])
                
                if min_dist_to_obstacles > 1.5:
                    self.goal_x = new_x
                    self.goal_y = new_y
                    safe_goal_found = True
                
                attempts += 1
            
            if not safe_goal_found:
                safe_positions = [(2.0, 2.0), (9.0, 9.0), (2.0, 9.0), (9.0, 2.0)]
                goal_pos = safe_positions[np.random.randint(0, len(safe_positions))]
                self.goal_x, self.goal_y = goal_pos
            
            self.get_logger().info(f'🎯 Nuevo objetivo: ({self.goal_x:.2f}, {self.goal_y:.2f})')
            self.draw_goal()
        
        # Publicar comandos
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.velocity_publisher.publish(cmd)
        
        # Log periódico
        self.log_counter += 1
        if self.log_counter % 20 == 0:
            status = "🚧 EVITANDO" if dist_to_obstacle < 2.0 else "✅ LIBRE"
            self.get_logger().info(
                f'{status} | Pos: ({self.current_pose.x:.1f}, {self.current_pose.y:.1f}) | '
                f'Dist Obj: {distance_to_goal:.2f} | Dist Obs: {dist_to_obstacle:.2f} | '
                f'Vel: ({linear_vel:.2f}, {angular_vel:.2f})')


def main(args=None):
    rclpy.init(args=args)
    controller = TurtleNNController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('🛑 Deteniendo...')
    finally:
        try:
            cmd = Twist()
            controller.velocity_publisher.publish(cmd)
        except:
            pass
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

