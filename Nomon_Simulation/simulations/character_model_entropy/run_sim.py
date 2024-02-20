#!/usr/bin/python

import os, sys, inspect
import numpy as np
import pandas as pd
from datetime import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
maindir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
os.chdir(parentdir)

from Nomon_Simulation.simulated_user_text import SimulatedUser

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
        self.date_time_str = now.strftime("%m_%d_%Y-%H_%M")
        self.output_dir = currentdir + "/results/sim-" + self.date_time_str + "/"

    def save_results(self, result_df, my_task_id, job):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        save_filename = self.output_dir + 'data-task_' + str(my_task_id) + '-job_' + str(job) + '.csv'
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
            # remove session num from clicks to specify as calibration data
            symbol_click_df['Session Num'] = np.nan
            symbol_click_df['Dead Time (s)'] = np.nan
        else:
            symbol_click_df = pd.DataFrame(columns=["Session Num", "Clock Period (s)",
                                                    "Click Time Relative (s)", "Dead Time (s)"])

        # load text entry task data for use in simulation
        text_data_fname = os.path.dirname(parentdir) + "/Nomon_User_Data/OSF Data/text_entry_task/user_" + str(
            user_id) + "_text_click_data_clean.csv"
        text_click_df = pd.read_csv(text_data_fname, usecols=["Session Num", "Clock Period (s)",
                                                              "Click Time Relative (s)", "Dead Time (s)"])

        full_click_df = pd.concat([symbol_click_df, text_click_df])

        print("User " + str(user_id) + ": Loaded " + str(symbol_click_df.shape[0]) + " calibration clicks, and " + str(
            text_click_df.shape[0]) + " clicks!")
        return full_click_df

    def run_job(self, my_task_id, num_tasks):
        lm_types = ["tiny", "medium", "huge"]
        lm_crosses = [(char, word) for char in lm_types for word in lm_types]

        parameters_li = []
        for pair in lm_crosses:
            parameters_li.append({"char_lm": pair[0], "word_lm": pair[1]})

        parameters_li = [{"char_lm": "dec19", "word_lm": "dec19"}]

        # for lm_type in lm_types:
        #     parameters_li.append({"char_lm": lm_type, "word_lm": lm_type})


        task_jobs = np.array_split(parameters_li, num_tasks)[my_task_id - 1]

        print("Running jobs ", task_jobs, " of ", len(parameters_li))

        for job_num, job in enumerate(task_jobs):
            # initialize new sim
            sim = SimulatedUser()

            job_char_lm = job["char_lm"]
            job_word_lm = job["word_lm"]

            if job_char_lm == "tiny":
                char_lm_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'lm_char_tiny.kenlm')
            elif job_char_lm == "medium":
                char_lm_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'lm_char_medium.kenlm')
            elif job_char_lm == "huge":
                char_lm_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'lm_char_large.kenlm')
            elif job_word_lm == "dec19":
                char_lm_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'lm_char_dec19.kenlm')

            if job_word_lm == "tiny":
                word_lm_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'lm_word_tiny.kenlm')
            elif job_word_lm == "medium":
                word_lm_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'lm_word_medium.kenlm')
            elif job_word_lm == "huge":
                word_lm_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'lm_word_huge.kenlm')
            elif job_word_lm == "dec19":
                word_lm_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'lm_word_dec19.kenlm')

            vocab_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'vocab_lower_100k.txt')
            char_path = os.path.join(os.path.join(maindir, 'Nomon_Text/resources'), 'char_set.txt')

            user_id = "A"

            click_df = self.load_user_click_data(user_id)
            click_df = click_df[np.abs(click_df["Click Time Relative (s)"]) < 0.26]
            click_df = click_df[click_df["Dead Time (s)"] < 15]

            params = {"click_df": click_df,
                      "phrase_shuffle_seed": lambda trial: ord(user_id) + trial,
                      "lm_files": [word_lm_path, char_lm_path, vocab_path, char_path]}

            # run simulation with job parameters for specified number of trials
            sim.simulate_phrases(params, trials=1, verbose=False)

            # save results
            sim.result_df["user_id"] = user_id
            sim.result_df["char_lm_type"] = job_char_lm
            sim.result_df["word_lm_type"] = job_word_lm
            sim.result_df["phrase_set"] = "watch"
            sim.result_df["win_diff"] = 27.5
            self.save_results(sim.result_df, my_task_id, job_num)

            click_entropy_df = pd.DataFrame(sim.cumulative_cscores)
            self.output_dir = currentdir + "/click_results/sim-" + self.date_time_str + "/"
            self.save_results(click_entropy_df, my_task_id, job_num)

            # free up memory by deleting old sim
            del sim


def main():
    sim_util = SimulationUtil()
    sim_util.run_job(0, 1)


if __name__ == "__main__":
    main()
