import os,sys,inspect
import numpy as np
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
os.chdir(parentdir)
from Nomon_Simulation.simulated_user_symbol import SimulatedUser

try:
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
except IndexError:
    my_task_id = 1
    num_tasks = 1


class simulationUtil():
    def __init__(self):
        # load data
        data_dir = os.path.dirname(parentdir)+"\\Nomon_User_Data\\SE_Study\\user_84_click_data.csv"
        self.click_df = pd.read_csv(data_dir, usecols=["Session Num","Target","Clock Period (s)",
                                                  "Click Time Relative (s)","Dead Time (s)"])
        print("Loaded "+str(self.click_df.shape[0])+" clicks!")

    def run_job(self, my_task_id, num_tasks):
        parameters_li = [1]

        task_jobs = np.array_split(parameters_li, num_tasks)[my_task_id - 1]

        print(task_jobs)

        for i in task_jobs:
            sim = SimulatedUser()
            params = {"click_df": self.click_df}

            sim.parameter_metrics(params, trials=1)
            print(sim.result_df)

            # compression_opts = dict(method='zip',
            #                         archive_name='sim_gaze_'+str(gaze_scale)+'.csv')
            # sim.result_df.to_csv('simulations/ajay_data/eyegaze_normal_results/sim_gaze_'+str(gaze_scale)+'.zip', index=False,
            #           compression=compression_opts)


def main():
    sim_util = simulationUtil()
    sim_util.run_job(0, 1)


if __name__ == "__main__":
    main()

