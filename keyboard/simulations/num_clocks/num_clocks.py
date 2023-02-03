import os,sys,inspect
import numpy as np
import emoji
import csv
import re
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
    num_tasks = 4

with open('simulations/ajay_data/click_times_user_84.csv', newline='') as f:
    reader = csv.reader(f)
    user_data = list(reader)



class simulationUtil():
    def __init__(self):
        emoji_file = open("resources/emojis_long.txt", "r")
        emoji_text = emoji_file.read()
        emoji_file.close()
        emoji_keys = emoji_text.split("\n")
        self.emoji_keys = [emoji.emojize(key, use_aliases=True) for key in emoji_keys]

    def run_job(self, my_task_id, num_tasks):
        data_len = len(user_data)
        # click_dists = user_data[:data_len // 2]
        # delay_dists = user_data[data_len // 2:]
        #
        # click_dists = click_dists[-3:]
        # delay_dists = delay_dists[-3:]
        click_dists = [[10] + np.random.normal(0, 0.1, 200).tolist() for i in range(5)]
        delay_dists = [np.abs(np.random.normal(1, 0.5, 200)).tolist() for i in range(5)]

        num_clocks = np.hstack([np.arange(1, 10, 1), np.arange(10, 70*3, 5)])

        # win_diffs = np.array([0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
        # win_diffs = np.floor(1 / win_diffs)

        parameters_li = []
        for nc in num_clocks:
            parameters_li += [[nc]]

        np.random.seed(0)
        np.random.shuffle(parameters_li)

        parameters_li = np.array(self.remove_redundant_jobs(parameters_li))

        num_jobs = len(parameters_li)

        task_jobs = np.array_split(parameters_li, num_tasks)[my_task_id - 1]

        print(task_jobs, len(task_jobs), num_jobs)

        for job_num, params in enumerate(task_jobs):

            nc = params[0]

            custom_keys = self.emoji_keys[-3-int(nc):]

            sim = SimulatedUser(custom_keys=custom_keys, job_num=job_num, num_jobs=len(task_jobs))
            params = {"N_pred": 0, "num_words": 0, "time_rotate": 14, "gaze_scale": 0, "click_dist": click_dists,
                  "delay_dist": delay_dists}

            sim.parameter_metrics(params, trials=15)

            sim.result_df["num_clocks"] = nc+3

            print(sim.result_df)

            compression_opts = dict(method='zip',
                                    archive_name='sim_num_clocks_'+str(nc)+'.csv')
            sim.result_df.to_csv('simulations/num_clocks/normal_results/sim_num_clocks_'+str(nc)+'.zip', index=False,
                      compression=compression_opts)

    def remove_redundant_jobs(self, jobs):

        path = "simulations/num_clocks/normal_results/"

        dir_list = os.listdir(path)

        completed_tasks = set()
        for filename in dir_list:
            p = re.compile(r'sim_num_clocks*')
            if p.match(filename) is not None:
                p = re.compile(r'\d+')
                result = p.findall(filename)
                completed_tasks.add((int(result[0])))

        print(jobs)
        jobs = set(tuple(job) for job in jobs)
        return list(jobs-completed_tasks)



def main():
    sim_util = simulationUtil()
    result = sim_util.run_job(0, 2)


if __name__ == "__main__":
    main()