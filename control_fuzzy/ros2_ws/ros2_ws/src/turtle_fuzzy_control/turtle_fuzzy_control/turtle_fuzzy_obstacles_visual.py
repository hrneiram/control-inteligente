#!/usr/bin/env python3
"""
Control Difuso con Obstáculos Visualizados como Círculos
Dibuja círculos rojos para representar obstáculos
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import Spawn, Kill, TeleportAbsolute, SetPen
import math
import numpy as np
import time


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


class FuzzyObstacleVisualController(Node):
    def __init__(self):
        super().__init__('turtle_fuzzy_obstacle_visual')
        
        # Publisher para comandos de velocidad
        self.velocity_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Subscriber para la pose actual
        self.pose_subscriber = self.create_subscription(
            Pose, '/turtle1/pose', self.pose_callback, 10)
        
        # Variables de estado
        self.current_pose = Pose()
        
        # Posición inicial segura (abajo-izquierda, lejos de obstáculos)
        self.start_x = 2.0
        self.start_y = 2.0
        
        # Objetivo inicial (arriba-derecha)
        self.goal_x = 9.0
        self.goal_y = 9.0
        self.obstacles_drawn = False
        
        # Definir obstáculos (x, y, radio) - reposicionados para dejar espacio
        self.obstacles = [
            Obstacle(5.5, 5.5, 0.4),   # Centro
            Obstacle(3.5, 8.0, 0.4),   # Superior izquierda
            Obstacle(8.0, 3.5, 0.4),   # Inferior derecha
        ]
        
        self.get_logger().info('🎯 Control Difuso Visual iniciado!')
        self.get_logger().info('⏳ Moviendo tortuga a posición inicial segura...')
        
        # Mover tortuga a posición inicial
        self.move_turtle_to_start()
        
        self.get_logger().info('⏳ Dibujando obstáculos y objetivo...')
        
        # Timer para dibujar obstáculos (se ejecuta una vez)
        self.draw_timer = self.create_timer(1.5, self.draw_obstacles_once)
        
        # Timer para control (se inicia después)
        self.control_timer = None
    
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
        self.get_logger().info('🚧 Obstáculos y objetivo dibujados! Iniciando navegación...')
        self.get_logger().info(f'📍 Objetivo inicial: ({self.goal_x}, {self.goal_y})')
    
    def draw_obstacles(self):
        """Dibuja círculos rojos para representar obstáculos"""
        for i, obs in enumerate(self.obstacles):
            # Crear una tortuga temporal para dibujar
            spawn_name = f'drawer_{i}'
            
            # Spawn tortuga
            self.spawn_turtle(spawn_name, obs.x + obs.radius, obs.y)
            time.sleep(0.3)
            
            # Configurar pen (rojo, grosor 3)
            self.set_pen(spawn_name, 255, 0, 0, 3, 0)  # off=0 para activar
            time.sleep(0.1)
            
            # Dibujar círculo
            self.draw_circle(spawn_name, obs.radius)
            time.sleep(0.3)
            
            # Eliminar tortuga temporal
            self.kill_turtle(spawn_name)
            time.sleep(0.1)
    
    def draw_goal(self):
        """Dibuja un círculo verde pequeño en la posición del objetivo"""
        spawn_name = 'goal_drawer'
        goal_radius = 0.3  # Radio pequeño para el objetivo
        
        # Spawn tortuga en el borde del círculo del objetivo
        self.spawn_turtle(spawn_name, self.goal_x + goal_radius, self.goal_y)
        time.sleep(0.2)
        
        # Configurar pen (verde, grosor 2)
        self.set_pen(spawn_name, 0, 255, 0, 2, 0)
        time.sleep(0.1)
        
        # Dibujar círculo pequeño
        self.draw_circle(spawn_name, goal_radius)
        time.sleep(0.2)
        
        # Eliminar tortuga temporal
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
        
        # Calcular velocidades para círculo
        linear_speed = 1.0
        angular_speed = linear_speed / radius
        
        # Tiempo para completar círculo
        circle_time = 2 * math.pi * radius / linear_speed
        
        # Publicar comando
        cmd = Twist()
        cmd.linear.x = linear_speed
        cmd.angular.z = angular_speed
        
        start_time = time.time()
        
        while (time.time() - start_time) < circle_time:
            vel_pub.publish(cmd)
            time.sleep(0.02)
        
        # Detener
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
    
    def get_obstacle_avoidance_direction(self, x, y, angle_to_goal):
        """Calcula dirección de evitación mejorada"""
        closest_obs, dist = self.find_closest_obstacle(x, y)
        
        # Si está muy lejos, no hay peligro
        if dist > 2.5:
            return angle_to_goal, 0.0
        
        # Ángulo desde la tortuga hacia el obstáculo
        angle_to_obstacle = math.atan2(closest_obs.y - y, closest_obs.x - x)
        
        # Si está MUY cerca (casi tocando), huir directamente en dirección opuesta
        if dist < 0.5:
            escape_angle = angle_to_obstacle + math.pi  # Dirección opuesta
            return escape_angle, 1.0  # Urgencia máxima
        
        # Calcular ángulos perpendiculares para rodear
        perp_angle_1 = angle_to_obstacle + math.pi / 2
        perp_angle_2 = angle_to_obstacle - math.pi / 2
        
        # Elegir la dirección perpendicular más cercana al objetivo
        diff_1 = abs(math.atan2(math.sin(perp_angle_1 - angle_to_goal), 
                                 math.cos(perp_angle_1 - angle_to_goal)))
        diff_2 = abs(math.atan2(math.sin(perp_angle_2 - angle_to_goal), 
                                 math.cos(perp_angle_2 - angle_to_goal)))
        
        avoidance_angle = perp_angle_1 if diff_1 < diff_2 else perp_angle_2
        
        # Calcular urgencia (más cerca = más urgente)
        urgency = max(0, min(1, (2.5 - dist) / 2.5))
        
        return avoidance_angle, urgency
    
    def apply_fuzzy_control(self, distance, angle_error, avoidance_urgency):
        """Lógica difusa mejorada"""
        # Velocidad lineal basada en urgencia de evitación
        if avoidance_urgency > 0.9:
            # Urgencia crítica: retroceder o ir muy lento
            linear_vel = 0.15
        elif avoidance_urgency > 0.7:
            # Urgencia alta: muy lento
            linear_vel = 0.3
        elif avoidance_urgency > 0.4:
            # Urgencia media: lento
            linear_vel = 0.6
        elif distance < 1.0:
            # Cerca del objetivo
            linear_vel = 0.5
        elif distance < 3.0:
            # Distancia media
            linear_vel = 1.2
        else:
            # Lejos: velocidad normal
            linear_vel = 1.5
        
        # Velocidad angular basada en error de ángulo y urgencia
        angle_abs = abs(angle_error)
        
        if avoidance_urgency > 0.7:
            # Giro urgente para evitar
            angular_vel = 2.5 * np.sign(angle_error)
        elif angle_abs > 1.5:
            # Muy desalineado
            angular_vel = 2.2 * np.sign(angle_error)
        elif angle_abs > 1.0:
            # Desalineado
            angular_vel = 1.8 * np.sign(angle_error)
        elif angle_abs > 0.5:
            # Poco desalineado
            angular_vel = 1.0 * np.sign(angle_error)
        else:
            # Casi alineado
            angular_vel = 0.5 * np.sign(angle_error)
        
        # Reducir velocidad lineal proporcionalmente a la urgencia
        if avoidance_urgency > 0.3:
            linear_vel *= (1 - avoidance_urgency * 0.8)
        
        return linear_vel, angular_vel
    
    def control_loop(self):
        """Bucle principal de control"""
        if not self.obstacles_drawn:
            return
        
        # Calcular objetivo
        dx = self.goal_x - self.current_pose.x
        dy = self.goal_y - self.current_pose.y
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_goal = math.atan2(dy, dx)
        
        # Evitación de obstáculos
        avoidance_angle, avoidance_urgency = self.get_obstacle_avoidance_direction(
            self.current_pose.x, self.current_pose.y, angle_to_goal)
        
        # Mezclar ángulos - dar más prioridad a evitación cuando hay urgencia
        if avoidance_urgency > 0.8:
            # Urgencia muy alta: ignorar objetivo completamente
            target_angle = avoidance_angle
        elif avoidance_urgency > 0.4:
            # Urgencia media: mezclar con más peso en evitación
            weight = avoidance_urgency ** 2  # Peso cuadrático para mayor énfasis
            target_angle = avoidance_angle * weight + angle_to_goal * (1 - weight)
        else:
            # Baja urgencia: priorizar objetivo
            target_angle = angle_to_goal
        
        angle_error = target_angle - self.current_pose.theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        # Verificar llegada
        if distance < 0.5:
            self.get_logger().info('🎉 ¡Objetivo alcanzado!')
            
            # Generar nuevo objetivo en posición segura (lejos de obstáculos)
            safe_goal_found = False
            attempts = 0
            while not safe_goal_found and attempts < 20:
                new_x = np.random.uniform(2.0, 9.0)
                new_y = np.random.uniform(2.0, 9.0)
                
                # Verificar que esté lejos de todos los obstáculos
                min_dist_to_obstacles = min([obs.distance_to(new_x, new_y) for obs in self.obstacles])
                
                if min_dist_to_obstacles > 1.5:  # Al menos 1.5 unidades de los obstáculos
                    self.goal_x = new_x
                    self.goal_y = new_y
                    safe_goal_found = True
                
                attempts += 1
            
            if not safe_goal_found:
                # Fallback: usar posiciones predefinidas seguras
                safe_positions = [(2.0, 2.0), (9.0, 9.0), (2.0, 9.0), (9.0, 2.0)]
                goal_pos = safe_positions[np.random.randint(0, len(safe_positions))]
                self.goal_x, self.goal_y = goal_pos
            
            self.get_logger().info(f'🎯 Nuevo objetivo: ({self.goal_x:.2f}, {self.goal_y:.2f})')
            
            # Redibujar el objetivo
            self.draw_goal()
        
        # Control difuso
        linear_vel, angular_vel = self.apply_fuzzy_control(
            distance, angle_error, avoidance_urgency)
        
        # Publicar
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.velocity_publisher.publish(cmd)
        
        # Log
        if not hasattr(self, 'log_counter'):
            self.log_counter = 0
        self.log_counter += 1
        
        if self.log_counter % 20 == 0:
            closest_obs, obs_dist = self.find_closest_obstacle(
                self.current_pose.x, self.current_pose.y)
            
            status = "🚧 EVITANDO" if avoidance_urgency > 0.5 else "✅ LIBRE"
            self.get_logger().info(
                f'{status} | Pos: ({self.current_pose.x:.1f}, {self.current_pose.y:.1f}) | '
                f'Dist Obj: {distance:.2f} | Dist Obs: {obs_dist:.2f} | Vel: {linear_vel:.2f}')


def main(args=None):
    rclpy.init(args=args)
    controller = FuzzyObstacleVisualController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('🛑 Deteniendo...')
    finally:
        # Detener solo si el contexto aún es válido
        try:
            cmd = Twist()
            controller.velocity_publisher.publish(cmd)
        except:
            pass
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
