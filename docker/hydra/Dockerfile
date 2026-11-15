# Hydra (MIT-SPARK) build image.
# NOTE (WP1 step 1): verify the ROS distro and workspace setup against the current
# upstream hydra_ros README before the first build, then pin HYDRA_ROS_REF to a commit.
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ARG HYDRA_ROS_REF=main
ENV DEBIAN_FRONTEND=noninteractive \
    WS=/ws

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3-pip python3-vcstool python3-colcon-common-extensions \
        python3-rosdep build-essential cmake \
        ros-${ROS_DISTRO}-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# Fetch hydra_ros and its dependencies (repos file name: check upstream README)
WORKDIR ${WS}/src
RUN git clone https://github.com/MIT-SPARK/Hydra-ROS.git hydra_ros \
    && cd hydra_ros && git checkout ${HYDRA_ROS_REF} \
    && cd .. && vcs import . < hydra_ros/install/ros2.repos

WORKDIR ${WS}
RUN apt-get update \
    && (rosdep init || true) && rosdep update \
    && rosdep install --from-paths src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/*

RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo ". /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc \
    && echo ". ${WS}/install/setup.bash" >> /root/.bashrc

WORKDIR /project
CMD ["bash"]
