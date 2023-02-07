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
        data_dir = os.path.dirname(parentdir)+"/Nomon_User_Data/SE_Study/user_84_click_data.csv"
        self.click_df = pd.read_csv(data_dir, usecols=["Session Num","Target","Clock Period (s)",
                                                  "Click Time Relative (s)","Dead Time (s)"])
        print("Loaded "+str(self.click_df.shape[0])+" clicks!")

        self.output_dir = parentdir+"/simulations/mi_user_sim/results/"

    def run_job(self, my_task_id, num_tasks):
        parameters_li = [1]

        task_jobs = np.array_split(parameters_li, num_tasks)[my_task_id - 1]

        print("Running jobs ", task_jobs, " of ", len(parameters_li))

        for job in task_jobs:
            sim = SimulatedUser()
            params = {"click_df": self.click_df}

            sim.parameter_metrics(params, trials=10, verbose=False)
            print(sim.result_df)

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            save_filename = self.output_dir + 'sim_data_'+str(job)+'.csv'
            print("Saving to: "+self.output_dir)
            sim.result_df.to_csv(save_filename, index=False)


def main():
    sim_util = simulationUtil()
    sim_util.run_job(0, 1)


if __name__ == "__main__":
    main()

