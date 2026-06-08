#!/usr/bin/python

import os,sys,inspect
import numpy as np
import pandas as pd
from datetime import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
os.chdir(parentdir)
from User_Simulation.simulated_user_symbol import SimulatedUser

### code used for distributed computing
# my_task_id specifies the task (virtual cpu) running the current simulation
try:
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
except IndexError:
    my_task_id = 1
    num_tasks = 1


class SimulationUtil():
    def __init__(self):
        # specify where to save results
        now = datetime.now()
        date_time_str = now.strftime("%m_%d_%Y-%H_%M")
        self.output_dir = currentdir + "/results/sim-" + date_time_str + "/"

    def save_results(self, result_df, my_task_id, job):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        save_filename = self.output_dir + 'data-task_'+str(my_task_id)+'-job_' + str(job) + '.csv'
        print("Saving to: " + self.output_dir)
        result_df.to_csv(save_filename, index=False)

    def load_user_click_data(self, user_id):
        # load picture selection task click data for initial calibration
        symbol_data_fname = os.path.dirname(parentdir) + "/Nomon_User_Data/OSF Data/picture_selection_task/user_" + str(
            user_id) + "_click_data.csv"
        # check if user has data from picture selection task (user B does not)
        if os.path.exists(symbol_data_fname):
            symbol_click_df = pd.read_csv(symbol_data_fname, usecols=["Session Num", "Clock Period (s)",
                                                                "Click Time Relative (s)", "Dead Time (s)"])
        else:
            symbol_click_df = pd.DataFrame(columns=["Session Num", "Clock Period (s)",
                                                                "Click Time Relative (s)", "Dead Time (s)"])

        print("User " + str(user_id) + ": Loaded " + str(symbol_click_df.shape[0]) + " clicks.")
        return symbol_click_df

    def run_job(self, my_task_id, num_tasks):
        parameters_li = ["A", "C", "D", "E", "F", "G"]
        parameters_li = ["A"]

        task_jobs = np.array_split(parameters_li, num_tasks)[my_task_id - 1]

        print("Running jobs ", task_jobs, " of ", len(parameters_li))

        for job_num, job in enumerate(task_jobs):
            # initialize new sim
            sim = SimulatedUser()
            params = {"click_df": self.load_user_click_data(job)}

            # run simulation with job parameters for specified number of trials
            sim.parameter_metrics(params, trials=1, verbose=False)

            # save results
            sim.result_df["user_id"] = job
            self.save_results(sim.result_df, my_task_id, job_num)

            # free up memory by deleting old sim
            del sim


def main():
    sim_util = SimulationUtil()
    sim_util.run_job(0, 1)


if __name__ == "__main__":
    main()

