import pandas as pd

import datetime as dt

import numpy as np
import json
import os

import ast

from dav_kafka_python.producer import PythonProducer
from dav_kafka_python.configuration import Configuration


from SEAS.federate_agent import FederateAgent
import datetime as dt

LOGFILE = str(dt.datetime.now()).replace(
    ":", "_").replace(" ", "_").replace(".", "_")

class Emulator(FederateAgent):

    def __init__(self, controller, py_sims, input_dict):
        
        # Save the input dict
        self.input_dict = input_dict

        # Save timt step
        self.dt = input_dict['dt']

        # Initialize components
        self.controller = controller
        self.py_sims = py_sims

        # Update the input dict components
        self.input_dict['controller'] = self.controller.get_controller_dict()
        self.input_dict['py_sims'] = self.py_sims.get_py_sim_dict()

        # HELICS dicts
        self.emu_comms_dict = input_dict['emu_comms']
        self.emu_helics_dict = self.emu_comms_dict['helics']
        self.helics_config_dict = self.emu_comms_dict['helics']['config']

        # Write the time step into helics config dict
        self.helics_config_dict['helics']['deltat'] = self.dt

        # Initialize the Federate class for HELICS communitation
        super(Emulator, self).__init__(
            name=self.helics_config_dict['name'], 
            starttime=self.helics_config_dict['starttime'],
            endtime=self.helics_config_dict['stoptime'], 
            config_dict=self.helics_config_dict
        )

        # TODO: Store other things
        self.use_dash_frontend = self.helics_config_dict["use_dash_frontend"]
        self.KAFKA = self.helics_config_dict["KAFKA"]

        # TODO Copied direct from control_center.py but not actually ready yet
        if self.KAFKA:
            # Kafka topic :
            self.topic = self.helics_config_dict["KAFKA_TOPIC"]
            print("KAFKA topic", self.topic)
            config = Configuration(env_path='./.env')
            self.python_producer = PythonProducer(config)
            self.python_producer.connect()

        # AMR wind files
        # Grab py sim details
        self.amr_wind_dict = self.emu_comms_dict['amr_wind']

        self.n_amr_wind = len(self.amr_wind_dict )
        self.amr_wind_names = list(self.amr_wind_dict.keys())

        # Save information about amr_wind simulations
        for amr_wind_name in self.amr_wind_names:
            self.amr_wind_dict[amr_wind_name].update(
                self.read_amr_wind_input(
                    self.amr_wind_dict[amr_wind_name]['amr_wind_input_file']
                )
            )

        #TODO For now, need to assume for simplicity there is one and only
        # one AMR_Wind simualtion
        self.num_turbines = self.amr_wind_dict[self.amr_wind_names[0]]['num_turbines']
        self.rotor_diameter = self.amr_wind_dict[self.amr_wind_names[0]]['rotor_diameter']
        self.turbine_locations = self.amr_wind_dict[self.amr_wind_names[0]]['turbine_locations']
        self.turbine_labels = self.amr_wind_dict[self.amr_wind_names[0]]['turbine_labels']

        # TODO In fugure could cover multiple farms
        # Initialize the turbine power array
        self.turbine_power_array = np.zeros(self.num_turbines)
        self.amr_wind_dict[self.amr_wind_names[0]]['turbine_powers'] = np.zeros(self.num_turbines)

        #TODO Could set up logging here

        #TODO Set interface comms to either dash or kenny's front end

        #TODO Set comms to non-helics based things like http polling

        # TODO not positive if this is the right place but I think it is
        # Hold here and wait for AMR Wind to respond
        # Note we're passing a few intiial wind speed and direction things
        # but we can come back to all that

        # FORMER CODE
        # self.logger.info("... waiting for initial connection from AMRWind")
        # list(self.pub.values())[0].publish(str("[-1,-1,-1]"))
        # self.logger.info(" #### Entering main loop #### ")


    def run(self):

        #TODO In future code that doesnt insist on AMRWInd can make this optional
        print("... waiting for initial connection from AMRWind")
        # Send initial connection signal to AMRWind
        # publish on topic: control
        self.send_via_helics("control", str("[-1,-1,-1]"))
        print(" #### Entering main loop #### ")
            
        # Run simulation till  endtime
        while self.absolute_helics_time < self.endtime:

            # Loop till we reach simulation startime. 
            if (self.absolute_helics_time < self.starttime):
                continue

            # Update controller and py sims
            self.controller.step(self.input_dict)
            self.input_dict['controller'] = self.controller.get_controller_dict()
            self.py_sims.step(self.input_dict)
            self.input_dict['py_sims'] = self.py_sims.get_py_sim_dict()

            # Print the input dict
            print(self.input_dict)

            # Subscribe to helics messages:
            incoming_messages = self.helics_connector.get_all_waiting_messages()
            if incoming_messages != {}:
                subscription_value  = self.process_subscription_messages(incoming_messages)    
            else:
                print("Emulator: Did not receive subscription from AMRWind, setting everyhthing to 0.")
                subscription_value  = [0, 0, 0] + [0 for t in range(self.num_turbines)] + [0 for t in range(self.num_turbines)]

            #TODO Parse returns from AMRWind
            sim_time_s_amr_wind, wind_speed_amr_wind, wind_direction_amr_wind = subscription_value[
                :3]
            turbine_power_array = subscription_value[3:3+self.num_turbines]
            turbine_wd_array = subscription_value[3+self.num_turbines:]
            self.wind_speed = wind_speed_amr_wind
            self.wind_direction = wind_direction_amr_wind

            #TODO F-Strings
            print("=======================================")
            print("AMRWindTime:", sim_time_s_amr_wind)
            print("AMRWindSpeed:", wind_speed_amr_wind)
            print("AMRWindDirection:", wind_direction_amr_wind)
            print("AMRWindTurbinePowers:", turbine_power_array)
            print(" AMRWIND number of turbines here: ", self.num_turbines)
            print("AMRWindTurbineWD:", turbine_wd_array)
            print("=======================================")

            #Process periocdic functions. 
            self.process_periodic_publication() 

            if self.KAFKA:
                key = json.dumps({"key": "wind_tower"})
                value = json.dumps({"helics_time": self.absolute_helics_time, "bucket": "wind_tower", "AMRWind_speed": wind_speed_amr_wind,
                                    "AMRWind_direction": wind_direction_amr_wind, "AMRWind_time": sim_time_s_amr_wind})
                self.python_producer.write(key=key, value=value,
                                            topic=self.topic, token='test-token')

            # Store turbine powers back to the dict
            #TODO hard-coded for now assuming only one AMR-WIND
            self.amr_wind_dict[self.amr_wind_names[0]]['turbine_powers'] = turbine_power_array
            self.turbine_power_array = turbine_power_array

            self.sync_time_helics(self.absolute_helics_time + self.deltat)


    def parse_input_yaml(self, filename):
        pass


    def process_subscription_messages(self, msg):
        # process data from HELICS subscription
        print(
            f"{self.name}, {self.absolute_helics_time} subscribed to message {msg}", flush=True)
        try:
            return list(ast.literal_eval(msg["status"]["message"]))
        except Exception as e:
            print(f"Subscription error:  {e} , returning 0s ", flush=True)
            return [0, 0, 0] + [0 for t in range(self.num_turbines)] + [0 for t in range(self.num_turbines)]

    def process_periodic_publication(self):
        # Periodically publish data to the surrpogate

        # self.get_signals_from_front_end()
        # self.set_wind_speed_direction()

        #yaw_angles = [270 for t in range(self.num_turbines)]
        yaw_angles = [270 for t in range(self.num_turbines)]
        # log these in kafka
        #yaw_angles[1] = 260

        # Send timing and yaw information to AMRWind via helics
        # publish on topic: control
        tmp = np.array([self.absolute_helics_time, self.wind_speed,
                       self.wind_direction] + yaw_angles).tolist()

        self.send_via_helics("control", str(tmp))


    def process_endpoint_event(self, msg):
        pass

    def process_periodic_endpoint(self):
        pass

    def read_amr_wind_input(self, amr_wind_input):

        # TODO this function is ugly and uncommented

        #TODO Initialize to empty in case doesn't run
        # Probably want a file not found error instead
        return_dict = {}

        with open(amr_wind_input) as fp:
            Lines = fp.readlines()

            # Find the actuators
            for line in Lines:
                if 'Actuator.labels' in line:
                    turbine_labels = line.split()[2:]
                    num_turbines = len(turbine_labels)

            self.num_turbines = num_turbines
            print("Number of turbines in amrwind: ", num_turbines)
            
            aa = [f"power_{i}" for i in range(num_turbines)]
            xyz = ",".join(aa)
            bb = [f"turbine_wd_direction_{i}" for i in range(
                num_turbines)]
            zyx = ",".join(bb)
            with open(f'{LOGFILE}.csv', 'a') as filex:
                filex.write('helics_time' + ',' + 'AMRwind_time' + ',' +
                            'AMRWind_speed' + ',' + 'AMRWind_direction' + ',' + xyz + ',' + zyx + os.linesep)

            # Find the diameter
            for line in Lines:
                if 'rotor_diameter' in line:
                    D = float(line.split()[-1])

            # Get the turbine locations
            turbine_locations = []
            for label in turbine_labels:
                for line in Lines:
                    if 'Actuator.%s.base_position' % label in line:
                        locations = tuple([float(f)
                                          for f in line.split()[-3:-1]])
                        turbine_locations.append(locations)
        
            return_dict = {
                'num_turbines':num_turbines,
                'turbine_labels':turbine_labels,
                'rotor_diameter':D,
                'turbine_locations':turbine_locations
            }

            # print(return_dict)
        return return_dict

