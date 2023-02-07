import os,sys,inspect
import numpy as np
import csv
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
os.chdir(parentdir)

from simulated_user_symbol import SimulatedUser, normal_hist
from matplotlib import pyplot as plt

try:
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
except IndexError:
    my_task_id = 1
    num_tasks = 1

with open('simulations/ajay_data/click_times_user_84.csv', newline='') as f:
    reader = csv.reader(f)
    user_data = list(reader)



class simulationUtil():
    def __init__(self):
        return


    def run_job(self, my_task_id, num_tasks):
        data_len = len(user_data)
        # click_dists = user_data[:data_len // 2]
        # delay_dists = user_data[data_len // 2:]
        #
        # click_dists = click_dists[-3:]
        # delay_dists = delay_dists[-3:]
        click_dists = [[10] + np.random.normal(0, 0.1, 200).tolist() for i in range(5)]
        delay_dists = [np.abs(np.random.normal(1, 0.5, 200)).tolist() for i in range(5)]

        parameters_li = np.hstack([np.arange(0, 5, 1), np.arange(5, 70, 5)])

        num_jobs = len(parameters_li)

        task_jobs = np.array_split(parameters_li, num_tasks)[my_task_id - 1]

        print(task_jobs)

        for gaze_scale in task_jobs:

            sim = SimulatedUser()
            params = {"N_pred": 0, "num_words": 0, "time_rotate": 14, "gaze_scale": gaze_scale, "click_dist": click_dists,
                  "delay_dist": delay_dists}

            sim.parameter_metrics(params, trials=100)

            sim.result_df["gaze_scale"] = gaze_scale

            print(sim.result_df)

            compression_opts = dict(method='zip',
                                    archive_name='sim_gaze_'+str(gaze_scale)+'.csv')
            sim.result_df.to_csv('simulations/ajay_data/eyegaze_normal_results/sim_gaze_'+str(gaze_scale)+'.zip', index=False,
                      compression=compression_opts)

def main():
    sim_util = simulationUtil()
    result = sim_util.run_job(0, 2)


if __name__ == "__main__":
    main()

