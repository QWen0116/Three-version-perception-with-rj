"""
The safety manager is used to collect the AV's hazard status and give the
control back to human if necessary
"""
import numpy as np
import carla


from opencda.core.safety.sensors import CollisionSensor, \
    TrafficLightDector, StuckDetector, OffRoadDetector
import openpyxl
import pandas as pd
from opencda.scenario_testing.evaluations.utils import lprint
import os

class SafetyManager:
    """
    A class that manages the safety of a given vehicle in a simulation environment.

    Parameters
    ----------
    vehicle: carla.Actor
        The vehicle that the SafetyManager is responsible for.
    params: dict
        A dictionary of parameters that are used to configure the SafetyManager.
    """
    def __init__(self, vehicle, params,carla_world=None):
        self.vehicle = vehicle
        self.print_message = params['print_message']
        self.sensors = [CollisionSensor(vehicle, params['collision_sensor']),
                        StuckDetector(params['stuck_dector']),
                        OffRoadDetector(params['offroad_dector']),
                        TrafficLightDector(params['traffic_light_detector'],
                                           vehicle)]
        self.carla_world = carla_world if carla_world is not None \
            else self.vehicle.get_world()

    def get_carla_sim_time(self):
        world = self.carla_world
        timestamp = world.get_snapshot().timestamp
        return timestamp.elapsed_seconds


    status_records = []
    def update_info(self, data_dict) -> dict:
        status_dict = {}
        for sensor in self.sensors:
            sensor.tick(data_dict)
            status_dict.update(sensor.return_status())

        SafetyManager.status_records.append(status_dict)

        if self.print_message:
            print_flag = False
            # only print message when it has hazard
            for key, val in status_dict.items():
                if val == True:
                    print_flag = True
                    break
            if print_flag:
                # print("Safety Warning from the safety manager:")
                print(f'{self.get_carla_sim_time()}s: {status_dict}')
                # logfile_infor = os.path.join('C:/Users/LabSD2/OpenCDA/evaluation_outputs/running_log', 'try1_'+'log.txt')
                # lprint(logfile_infor, f'{self.get_carla_sim_time()}s: {status_dict}')


    @classmethod
    def save_status_records_to_excel(cls, file_path):
        df = pd.DataFrame(cls.status_records)
        df.to_excel(file_path, index=False)


    def destroy(self):
        for sensor in self.sensors:
            sensor.destroy()


