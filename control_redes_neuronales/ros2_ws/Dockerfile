# Dockerfile para ROS2 humble con turtleSim y PyTorch para redes neuronales
FROM osrf/ros:humble-desktop-full

# Evitar prompts interactivos durante la instalacion
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar e instalar dependencias
RUN apt-get update && apt-get install -y \
    ros-humble-turtlesim \
    ros-humble-rqt* \
    python3-pip \
    python3-numpy \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Instalar PyTorch (CPU version, ligero para el contenedor)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias adicionales para redes neuronales
RUN pip3 install numpy matplotlib scikit-learn

# Configurar el entorno ROS2
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Crear un workspace para proyectos futuros
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Configurar display para GUI
ENV DISPLAY=:0

# Comando por defecto
CMD ["/bin/bash"]

