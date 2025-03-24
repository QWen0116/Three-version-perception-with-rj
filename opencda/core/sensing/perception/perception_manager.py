# -*- coding: utf-8 -*-
"""
Perception module base.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import weakref
import sys
import time
import threading
import carla
import cv2
import numpy as np
import open3d as o3d
import torch
import subprocess
import os
from opencda.scenario_testing.evaluations.utils import lprint
from datetime import datetime, timedelta
import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.common.misc import \
    cal_distance_angle, get_speed, get_speed_sumo
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import TrafficLight
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, \
    o3d_camera_lidar_fusion
import random
import psutil


class CameraSensor:
    """
    Camera manager for vehicle or infrastructure.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    relative_position : str
        Indicates the sensor is a front or rear camera. option:
        front, left, right.

    Attributes
    ----------
    image : np.ndarray
        Current received rgb image.
    sensor : carla.sensor
        The carla sensor that mounts at the vehicle.

    """

    def __init__(self, vehicle, world, relative_position, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('fov', '100')

        spawn_point = self.spawn_point_estimation(relative_position,
                                                  global_position)

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.image = None
        # timstamp?
        self.timstamp = None  
        self.frame = 0
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CameraSensor._on_rgb_image_event(
                weak_self, event))

        # camera attributes
        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

    @staticmethod
    def spawn_point_estimation(relative_position, global_position):

        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)
        x, y, z, yaw = relative_position

        # this is for rsu. It utilizes global position instead of relative
        # position to the vehicle
        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])
            pitch = -35

        carla_location = carla.Location(x=carla_location.x + x,
                                        y=carla_location.y + y,
                                        z=carla_location.z + z)

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    @staticmethod
    def _on_rgb_image_event(weak_self, event):
        """CAMERA  method"""
        self = weak_self()
        if not self:
            return
        image = np.array(event.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        # we need to remove the alpha channel
        image = image[:, :, :3]

        self.image = image
        self.frame = event.frame
        self.timestamp = event.timestamp


class LidarSensor:
    """
    Lidar sensor manager.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.

    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))
        blueprint.set_attribute(
            'dropoff_general_rate', str(
                config_yaml['dropoff_general_rate']))
        blueprint.set_attribute(
            'dropoff_intensity_limit', str(
                config_yaml['dropoff_intensity_limit']))
        blueprint.set_attribute(
            'dropoff_zero_intensity', str(
                config_yaml['dropoff_zero_intensity']))
        blueprint.set_attribute(
            'noise_stddev', str(
                config_yaml['noise_stddev']))

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))
        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.data = None
        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LidarSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Lidar  method"""
        self = weak_self()
        if not self:
            return

        # retrieve the raw lidar data and reshape to (N, 4)
        data = np.copy(np.frombuffer(event.raw_data, dtype=np.dtype('f4')))
        # (x, y, z, intensity)
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


class SemanticLidarSensor:
    """
    Semantic lidar sensor manager. This class is used when data dumping
    is needed.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.


    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = \
            world.get_blueprint_library(). \
                find('sensor.lidar.ray_cast_semantic')

        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.points = None
        self.obj_idx = None
        self.obj_tag = None

        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: SemanticLidarSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Semantic Lidar  method"""
        self = weak_self()
        if not self:
            return

        # shape:(n, 6)
        data = np.frombuffer(event.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32),
            ('ObjTag', np.uint32)]))

        # (x, y, z, intensity)
        self.points = np.array([data['x'], data['y'], data['z']]).T
        self.obj_tag = np.array(data['ObjTag'])
        self.obj_idx = np.array(data['ObjIdx'])

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


def get_yolo_cpu_usage():
    process = psutil.Process(os.getpid())
    return process.cpu_percent(interval=5)/ psutil.cpu_count(logical=True)  

def get_yolo_gpu_utilization():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
        )
        return int(output.decode('utf-8').strip())
    except Exception as e:
        return "N/A"

def monitor_system():
    while True:
        cpu_usage = get_yolo_cpu_usage()
        gpu_utilization = get_yolo_gpu_utilization()

        log_file_path = os.path.join('C:/Users/LabSD2/OpenCDA/evaluation_outputs/overhead_log', 'cpu_'+'log.txt')
        with open(log_file_path, 'a') as file:
            file.write(str(cpu_usage)+'\n')

        log_file_path = os.path.join('C:/Users/LabSD2/OpenCDA/evaluation_outputs/overhead_log', 'gpu_'+'log.txt')
        with open(log_file_path, 'a') as file:
            file.write(str(gpu_utilization)+'\n')

        time.sleep(5) 


class PerceptionManager:
    """
    Default perception module. Currenly only used to detect vehicles.

    Parameters
    ----------
    vehicle : carla.Vehicle
        carla Vehicle, we need this to spawn sensors.

    config_yaml : dict
        Configuration dictionary for perception.

    cav_world : opencda object
        CAV World object that saves all cav information, shared ML model,
         and sumo2carla id mapping dictionary.

    data_dump : bool
        Whether dumping data, if true, semantic lidar will be spawned.

    carla_world : carla.world
        CARLA world, used for rsu.

    Attributes
    ----------
    lidar : opencda object
        Lidar sensor manager.

    rgb_camera : opencda object
        RGB camera manager.

    o3d_vis : o3d object
        Open3d point cloud visualizer.
    """

    def __init__(self, vehicle, config_yaml, cav_world,
                 data_dump=False, carla_world=None, infra_id=None):


        self.prev_sim_time=None
        self.prev_real_time=None
        self.vehicle = vehicle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        # monitor_thread.start()

        self.carla_world = carla_world if carla_world is not None \
            else self.vehicle.get_world()
        self._map = self.carla_world.get_map()
        self.id = infra_id if infra_id is not None else vehicle.id

        self.activate = config_yaml['activate']
        self.camera_visualize = config_yaml['camera']['visualize']
        self.camera_num = config_yaml['camera']['num']
        self.lidar_visualize = config_yaml['lidar']['visualize']
        self.global_position = config_yaml['global_position'] \
            if 'global_position' in config_yaml else None

        self.cav_world = weakref.ref(cav_world)()
        ml_manager = cav_world.ml_manager


        if self.activate and data_dump:
            sys.exit("When you dump data, please deactivate the "
                     "detection function for precise label.")

        if self.activate and not ml_manager:
            sys.exit(
                'If you activate the perception module, '
                'then apply_ml must be set to true in'
                'the argument parser to load the detection DL model.')
        self.ml_manager = ml_manager

        self.initialize_model_management()

        # we only spawn the camera when perception module is activated or
        # camera visualization is needed
        if self.activate or self.camera_visualize:
            self.rgb_camera = []
            mount_position = config_yaml['camera']['positions']
            assert len(mount_position) == self.camera_num, \
                "The camera number has to be the same as the length of the" \
                "relative positions list"

            for i in range(self.camera_num):
                self.rgb_camera.append(
                    CameraSensor(
                        vehicle, self.carla_world, mount_position[i],
                        self.global_position))

        else:
            self.rgb_camera = None

        # we only spawn the LiDAR when perception module is activated or lidar
        # visualization is needed
        if self.activate or self.lidar_visualize:
            self.lidar = LidarSensor(vehicle,
                                     self.carla_world,
                                     config_yaml['lidar'],
                                     self.global_position)
            self.o3d_vis = o3d_visualizer_init(self.id)
        else:
            self.lidar = None
            self.o3d_vis = None

        # if data dump is true, semantic lidar is also spawned
        self.data_dump = data_dump
        if data_dump:
            self.semantic_lidar = SemanticLidarSensor(vehicle,
                                                      self.carla_world,
                                                      config_yaml['lidar'],
                                                      self.global_position)

        # count how many steps have been passed
        self.count = 0
        # ego position
        self.ego_pos = None

        # the dictionary contains all objects
        self.objects = {}
        # traffic light detection related
        self.traffic_thresh = config_yaml['traffic_light_thresh'] \
            if 'traffic_light_thresh' in config_yaml else 50

    def measure_fps(self):
        sim_time = self.get_carla_sim_time()  
        real_time = time.time()  

        if self.prev_sim_time is not None and self.prev_real_time is not None:
            sim_fps = 1.0 / (sim_time - self.prev_sim_time) if sim_time - self.prev_sim_time > 0 else float('inf')
            real_fps = 1.0 / (real_time - self.prev_real_time) if real_time - self.prev_real_time > 0 else float('inf')
            # print(f"Simulation FPS: {sim_fps:.2f}, Real FPS: {real_fps:.2f}")

            log_file_path = os.path.join('C:/Users/LabSD2/OpenCDA/evaluation_outputs/overhead_log', 'fps_'+'log.txt')
            with open(log_file_path, 'a') as file:
                file.write(str(real_fps) + '\n')
        self.prev_sim_time = sim_time
        self.prev_real_time = real_time


    def dist(self, a):
        """
        A fast method to retrieve the obstacle distance the ego
        vehicle from the server directly.

        Parameters
        ----------
        a : carla.actor
            The obstacle vehicle.

        Returns
        -------
        distance : float
            The distance between ego and the target actor.
        """
        return a.get_location().distance(self.ego_pos.location)

    def detect(self, ego_pos):
        """
        Detect surrounding objects. Currently only vehicle detection supported.

        Parameters
        ----------
        ego_pos : carla.Transform
            Ego vehicle pose.

        Returns
        -------
        objects : list
            A list that contains all detected obstacle vehicles.

        """
        self.ego_pos = ego_pos

        objects = {'vehicles': [],
                   'traffic_lights': []}

        if not self.activate:
            objects = self.deactivate_mode(objects)

        else:
            objects = self.activate_mode(objects)

        self.count += 1

        # self.measure_fps()

        return objects

    # ****************
    def detections_to_numpy(self,detection):

        if detection.is_cuda:
            output = detection.cpu().detach().numpy()
        else:
            output = detection.detach().numpy()
        return output

    def get_carla_sim_time(self):
        world = self.carla_world
        timestamp = world.get_snapshot().timestamp
        return timestamp.elapsed_seconds 

    def count_model_states(self):
        healthy_count = sum(1 for status in self.healthyflag.values() if status == 2)
        compromised_count = sum(1 for status in self.healthyflag.values() if status == 1)
        failed_count = sum(1 for status in self.healthyflag.values() if status == 0)
        return healthy_count, compromised_count, failed_count

    #*********************destory and repair
    def update_models(self):
        tf=self.tflist[self.f]
        tc=self.tclist[self.c]
        current_time = self.get_carla_sim_time()

        for model_key in list(self.faulty_start_time.keys()):
            if (current_time - self.faulty_start_time[model_key]) >= tf:
                #fail
                self.healthyflag[model_key]=0
                h,c,f=self.count_model_states()
                print(f'{current_time}s: {h,c,f},model {model_key} failed, Tf={tf}')
                # lprint(self.logfile_infor, f'{current_time}s: {h,c,f}, model {model_key} failed, Tf={tf}')

                #  Restore the model to a healthy state
                self.current_models[model_key] = self.healthy_models[model_key]
                self.healthyflag[model_key]=2

                del self.faulty_start_time[model_key]
                #************mean time to recover (reactive rejuvenate)
                current_time_end=self.get_carla_sim_time()
                self.mean_time_to_recover=current_time_end-current_time
                # Compare mean_time_to_recover 
                wait_time = np.random.exponential(scale=0.5) #5
                if self.mean_time_to_recover > wait_time:
                    
                    h,c,f=self.count_model_states()
                    print(f'{self.get_carla_sim_time()}s: {h,c,f}, model {model_key} recovered, Tr={self.mean_time_to_recover}')
                    # lprint(self.logfile_infor, f'{self.get_carla_sim_time()}s: {h,c,f}, model {model_key} recovered, Tr={self.mean_time_to_recover}')

                else:
                 
                    time_to_wait = wait_time - self.mean_time_to_recover
                    time.sleep(time_to_wait)
                    h,c,f=self.count_model_states()
                    print(f'{self.get_carla_sim_time()}s: {h,c,f}, model {model_key} recovered, Tr={wait_time}')
                    # lprint(self.logfile_infor, f'{self.get_carla_sim_time()}s: {h,c,f}, model {model_key} recovered, Tr={wait_time}')

                self.f=self.f+1

            
        if (current_time - self.last_faulty_transition_time) >= tc:
            model_keys = list(self.healthy_models.keys())
            if self.current_transition_index < len(model_keys):
                model_to_fault = model_keys[self.current_transition_index]
                self.current_models[model_to_fault] = self.faulty_models['f' + model_to_fault[-1]]
                self.healthyflag[model_to_fault]=1
                self.faulty_start_time[model_to_fault] = current_time  
                self.current_transition_index = (self.current_transition_index + 1) % len(model_keys)
                self.last_faulty_transition_time = current_time
                h,c,f=self.count_model_states()
                print(f'{self.get_carla_sim_time()}s: {h,c,f}, model {model_to_fault} compromised, Tc={tc}')
                # lprint(self.logfile_infor, f'{self.get_carla_sim_time()}s: {h,c,f}, model {model_to_fault} compromised, Tc={tc}')
                self.c=self.c+1

                         
    def rejuvenate_models(self,t):
        current_time = self.get_carla_sim_time()
        if (current_time - self.last_refresh_time) >= t:

            num_faulty = sum(1 for model in self.current_models.values() if 'Faulty' in model.name)
            num_healthy = len(self.current_models) - num_faulty
            total = num_faulty + num_healthy

            if num_faulty > 0:
                prob_rejuvenate_faulty = 2/3 
            else:
                prob_rejuvenate_faulty = 0.0 

            if random.random() <= prob_rejuvenate_faulty:
                faulty_keys = [key for key, model in self.current_models.items() if 'Faulty' in model.name]
                if faulty_keys:
                    chosen_key = random.choice(faulty_keys)
                    self.current_models[chosen_key] = self.healthy_models[chosen_key]
                    self.healthyflag[chosen_key]=2

                    if chosen_key in self.faulty_start_time:
                        del self.faulty_start_time[chosen_key]
            else:
                healthy_keys = [key for key, model in self.current_models.items() if 'Healthy' in model.name]
                if healthy_keys:
                    chosen_key = random.choice(healthy_keys)

            self.last_refresh_time = current_time

            #*************************
            current_time_end=self.get_carla_sim_time()
            self.mean_time_to_rejuvenate=current_time_end-current_time
            wait_time = np.random.exponential(scale=0.5) #3.7
            if self.mean_time_to_rejuvenate > wait_time:
               
                h,c,f=self.count_model_states()
                print(f'{self.get_carla_sim_time()}s: {h,c,f}, model {chosen_key} rejuvenated, Trj={self.mean_time_to_rejuvenate}')
                # lprint(self.logfile_infor, f'{self.get_carla_sim_time()}s: {h,c,f}, model {chosen_key} rejuvenated, Trj={self.mean_time_to_rejuvenate}')
            else:
                
                time_to_wait = wait_time - self.mean_time_to_rejuvenate
                time.sleep(time_to_wait)
                h,c,f=self.count_model_states()
                print(f'{self.get_carla_sim_time()}s: {h,c,f}, model {chosen_key} rejuvenated, Trj={wait_time}')
                # lprint(self.logfile_infor, f'{self.get_carla_sim_time()}s: {h,c,f}, model {chosen_key} rejuvenated, Trj={wait_time}')

    def initialize_model_management(self):
        # Initializing models
        self.current_models = {
            'm1': self.ml_manager.object_detector1,
            'm2': self.ml_manager.object_detector2,
            'm3': self.ml_manager.object_detector3
        }
        self.healthy_models = self.current_models.copy()
        self.faulty_models = {
            'f1': self.ml_manager.object_detector4,
            'f2': self.ml_manager.object_detector5,
            'f3': self.ml_manager.object_detector6
        }
        self.healthyflag={
            'm1': 2,
            'm2': 2,
            'm3': 2,
        }
        self.last_faulty_transition_time = self.get_carla_sim_time()
        self.current_transition_index = 0
        self.last_refresh_time = self.get_carla_sim_time()
        self.faulty_start_time = {}  # To record the start time of faults
        self.mean_time_to_rejuvenate=0
        self.mean_time_to_recover=0
        self.timenow=datetime.now()
        np.random.seed(int(time.time()))
        self.tflist=np.random.exponential(scale=16, size=50) 
        self.tclist=np.random.exponential(scale=8, size=50)
        # print(self.tflist,self.tclist) 
        self.f=0
        self.c=0





    def activate_mode(self, objects):
        """
        Use Yolov5 + Lidar fusion to detect objects.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        # retrieve current cameras and lidar data


         
        rgb_images = []
        for rgb_camera in self.rgb_camera:
            while rgb_camera.image is None:
                continue
            rgb_images.append(
                cv2.cvtColor(
                    np.array(
                        rgb_camera.image),
                    cv2.COLOR_BGR2RGB))


        self.flag=1

        #***************************uodate/refresh funtion
        self.update_models()
        self.rejuvenate_models(t=3) 



        yolo_detection1=self.current_models['m1'].model(rgb_images)
        yolo_detection2=self.current_models['m2'].model(rgb_images)
        yolo_detection3=self.current_models['m3'].model(rgb_images)


        # rgb_images for drawing
        rgb_draw_images = []
        skipflagrecord = 0 
        recordflag=1

        for (i, rgb_camera) in enumerate(self.rgb_camera):
            # lidar projection
            rgb_image, projected_lidar = st.project_lidar_to_camera(
                self.lidar.sensor,
                rgb_camera.sensor, self.lidar.data, np.array(
                    rgb_camera.image))
            rgb_draw_images.append(rgb_image)


            if self.flag==1 or self.flag==3:
                #3-version
                detection1=yolo_detection1.xyxy[i]
                detection2=yolo_detection2.xyxy[i]
                detection3=yolo_detection3.xyxy[i]
                detections=[detection1,detection2,detection3]
                matched_detections=self.ml_manager.get_matched_detection(detections)
                final_yolo_detection,skipflag=self.ml_manager.get_final_detection(matched_detections)
                if recordflag==1:
                    skipflagrecord=skipflag
            elif self.flag==2:
            #     #2-version
                detection1=yolo_detection1.xyxy[i]
                detection2=yolo_detection2.xyxy[i]
                detections=[detection1,detection2]
                matched_detections=self.ml_manager.get_matched_detection(detections)
                final_yolo_detection,skipflag=self.ml_manager.get_final_detection(matched_detections)
                if recordflag==1:
                    skipflagrecord=skipflag   
            else:
                #1-version
                detection1=yolo_detection1.xyxy[i]
                final_yolo_detection=self.detections_to_numpy(yolo_detection1.xyxy[i])

            objects = o3d_camera_lidar_fusion(
                objects,
                final_yolo_detection,
                self.lidar.data,
                projected_lidar,
                self.lidar.sensor)

            # calculate the speed. current we retrieve from the server
            # directly.
            self.speed_retrieve(objects)




        if self.camera_visualize:
            for (i, rgb_image) in enumerate(rgb_draw_images):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break

                if self.flag==1 or self.flag==3:
                    #3-version
                    detection1=yolo_detection1.xyxy[i]
                    detection2=yolo_detection2.xyxy[i]
                    detection3=yolo_detection3.xyxy[i]
                    detections=[detection1,detection2,detection3]
                    matched_detections=self.ml_manager.get_matched_detection(detections)
                    final_yolo_detection,skipflag=self.ml_manager.get_final_detection(matched_detections)

                elif self.flag==2:
                #     #2-version
                    detection1=yolo_detection1.xyxy[i]
                    detection2=yolo_detection2.xyxy[i]
                    detections=[detection1,detection2]
                    matched_detections=self.ml_manager.get_matched_detection(detections)
                    final_yolo_detection,skipflag=self.ml_manager.get_final_detection(matched_detections)

                else:
                    #1-version
                    detection1=yolo_detection1.xyxy[i]
                    final_yolo_detection=self.detections_to_numpy(yolo_detection1.xyxy[i])


                label_names=yolo_detection1.names
                rgb_image = self.ml_manager.draw_2d_box(
                    final_yolo_detection, rgb_image,label_names)

                world = self.carla_world
                timestamp = world.get_snapshot().timestamp



                 # ************************************************************
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)
                cv2.imshow(
                    '%s-th camera of actor %d, perception activated' %
                    (str(i), self.id), rgb_image)

            cv2.waitKey(1)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)
        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

        return objects

    def deactivate_mode(self, objects):
        """
        Object detection using server information directly.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        world = self.carla_world

        vehicle_list = world.get_actors().filter("*vehicle*")
        # todo: hard coded
        thresh = 50 if not self.data_dump else 120

        vehicle_list = [v for v in vehicle_list if self.dist(v) < thresh and
                        v.id != self.id]

        # use semantic lidar to filter out vehicles out of the range
        if self.data_dump:
            vehicle_list = self.filter_vehicle_out_sensor(vehicle_list)

        # convert carla.Vehicle to opencda.ObstacleVehicle if lidar
        # visualization is required.
        if self.lidar:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    self.lidar.sensor,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]
        else:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    None,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]

        objects.update({'vehicles': vehicle_list})

        # ************************************************************

        if self.camera_visualize:
            while self.rgb_camera[0].image is None:
                continue

            names = ['front', 'right', 'left', 'back']

            for (i, rgb_camera) in enumerate(self.rgb_camera):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                # we only visualiz the frontal camera
                rgb_image = np.array(rgb_camera.image)
                # draw the ground truth bbx on the camera image
                rgb_image = self.visualize_3d_bbx_front_camera(objects,
                                                               rgb_image,
                                                               i)
                # resize to make it fittable to the screen
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)

                # show image using cv2
                cv2.imshow(
                    '%s camera of actor %d, perception deactivated' %
                    (names[i], self.id), rgb_image)
                cv2.waitKey(1)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            # render the raw lidar
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)

        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

        return objects

    def filter_vehicle_out_sensor(self, vehicle_list):
        """
        By utilizing semantic lidar, we can retrieve the objects that
        are in the lidar detection range from the server.
        This function is important for collect training data for object
        detection as it can filter out the objects out of the senor range.

        Parameters
        ----------
        vehicle_list : list
            The list contains all vehicles information retrieves from the
            server.

        Returns
        -------
        new_vehicle_list : list
            The list that filters out the out of scope vehicles.

        """
        semantic_idx = self.semantic_lidar.obj_idx
        semantic_tag = self.semantic_lidar.obj_tag

        # label 10 is the vehicle
        vehicle_idx = semantic_idx[semantic_tag == 10]
        # each individual instance id
        vehicle_unique_id = list(np.unique(vehicle_idx))

        new_vehicle_list = []
        for veh in vehicle_list:
            if veh.id in vehicle_unique_id:
                new_vehicle_list.append(veh)

        return new_vehicle_list

    def visualize_3d_bbx_front_camera(self, objects, rgb_image, camera_index):
        """
        Visualize the 3d bounding box on frontal camera image.

        Parameters
        ----------
        objects : dict
            The object dictionary.

        rgb_image : np.ndarray
            Received rgb image at current timestamp.

        camera_index : int
            Indicate the index of the current camera.

        """
        camera_transform = \
            self.rgb_camera[camera_index].sensor.get_transform()
        camera_location = \
            camera_transform.location
        camera_rotation = \
            camera_transform.rotation

        for v in objects['vehicles']:
            # we only draw the bounding box in the fov of camera
            _, angle = cal_distance_angle(
                v.get_location(), camera_location,
                camera_rotation.yaw)
            if angle < 60:
                bbx_camera = st.get_2d_bb(
                    v,
                    self.rgb_camera[camera_index].sensor,
                    camera_transform)
                cv2.rectangle(rgb_image,
                              (int(bbx_camera[0, 0]), int(bbx_camera[0, 1])),
                              (int(bbx_camera[1, 0]), int(bbx_camera[1, 1])),
                              (255, 0, 0), 2)

        return rgb_image

    def speed_retrieve(self, objects):
        """
        We don't implement any obstacle speed calculation algorithm.
        The speed will be retrieved from the server directly.

        Parameters
        ----------
        objects : dict
            The dictionary contains the objects.
        """
        if 'vehicles' not in objects:
            return

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 50 and
                        v.id != self.id]

        # todo: consider the minimum distance to be safer in next version
        for v in vehicle_list:
            loc = v.get_location()
            for obstacle_vehicle in objects['vehicles']:
                obstacle_speed = get_speed(obstacle_vehicle)
                # if speed > 0, it represents that the vehicle
                # has been already matched.
                if obstacle_speed > 0:
                    continue
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                        abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.set_velocity(v.get_velocity())

                    # the case where the obstacle vehicle is controled by
                    # sumo
                    if self.cav_world.sumo2carla_ids:
                        sumo_speed = \
                            get_speed_sumo(self.cav_world.sumo2carla_ids,
                                           v.id)
                        if sumo_speed > 0:
                            # todo: consider the yaw angle in the future
                            speed_vector = carla.Vector3D(sumo_speed, 0, 0)
                            obstacle_vehicle.set_velocity(speed_vector)

                    obstacle_vehicle.set_carla_id(v.id)

    def retrieve_traffic_lights(self, objects):
        """
        Retrieve the traffic lights nearby from the server  directly.
        Next version may consider add traffic light detection module.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all objects.

        Returns
        -------
        object : dict
            The updated dictionary.
        """
        world = self.carla_world
        tl_list = world.get_actors().filter('traffic.traffic_light*')

        vehicle_location = self.ego_pos.location
        vehicle_waypoint = self._map.get_waypoint(vehicle_location)

        activate_tl, light_trigger_location = \
            self._get_active_light(tl_list, vehicle_location, vehicle_waypoint)

        objects.update({'traffic_lights': []})

        if activate_tl is not None:
            traffic_light = TrafficLight(activate_tl,
                                         light_trigger_location,
                                         activate_tl.get_state())
            objects['traffic_lights'].append(traffic_light)
        return objects

    def _get_active_light(self, tl_list, vehicle_location, vehicle_waypoint):
        for tl in tl_list:
            object_location = \
                TrafficLight.get_trafficlight_trigger_location(tl)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != vehicle_waypoint.road_id:
                continue

            ve_dir = vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x +\
                        ve_dir.y * wp_dir.y + \
                        ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue
            while not object_waypoint.is_intersection:
                next_waypoint = object_waypoint.next(0.5)[0]
                if next_waypoint and not next_waypoint.is_intersection:
                    object_waypoint = next_waypoint
                else:
                    break

            return tl, object_waypoint.transform.location

        return None, None

    def destroy(self):
        """
        Destroy sensors.
        """
        if self.rgb_camera:
            for rgb_camera in self.rgb_camera:
                rgb_camera.sensor.destroy()

        if self.lidar:
            self.lidar.sensor.destroy()

        if self.camera_visualize:
            cv2.destroyAllWindows()

        if self.lidar_visualize:
            self.o3d_vis.destroy_window()

        if self.data_dump:
            self.semantic_lidar.sensor.destroy()
