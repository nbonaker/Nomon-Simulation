import os,sys,inspect
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
os.chdir(parentdir)

from simulated_user import SimulatedUser, normal_hist

try:
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
except IndexError:
    my_task_id = 40
    num_tasks = 82

total_job_descriptions = [(var_ind, rot_ind) for var_ind in range(1, 21) for rot_ind in range(0, 21)]
num_jobs = len(total_job_descriptions)
job_indices = np.array_split(np.arange(1, num_jobs), num_tasks)[my_task_id-1]

print(np.array(total_job_descriptions)[job_indices])

print([np.round(((0.5-0.01)/20*total_job_descriptions[i][0]+0.01), 3) for i in job_indices])

parameters_list = [{"click_dist": stats.norm(scale=np.round(((0.5-0.01)/20*total_job_descriptions[i][0]+0.01), 3)),
                    "time_rotate": total_job_descriptions[i-1][1],
                    "N_pred": 0} for i in job_indices]

attributes = [total_job_descriptions[i][0] for i in job_indices]
for parameters, attribute in zip(parameters_list, attributes):
    sim = SimulatedUser(currentdir)
    sim.parameter_metrics(parameters, num_clicks=2500, trials=50, attribute=attribute)
