######################################
# Copyright 2019 Nicholas Bonaker, Keith Vertanen, Emli-Mari Nel, Tamara Broderick
# This file is part of the Nomon software.
# Nomon is free software: you can redistribute it and/or modify it
# under the terms of the MIT License reproduced below.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
# OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR
#OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# <https://opensource.org/licenses/mit-license.html>
######################################


import os
from shutil import copyfile
from pickle_util import PickleUtil
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom
from scipy.interpolate import NearestNDInterpolator
import numpy as np
from row_col_sim_data_load import SimDataUtil_row_col


class SimDataUtil:

    def __init__(self, data_dir):
        self.data_directory = data_dir
        self.data_by_user = self.load_data()
        self.user_numbers = set(self.data_by_user.keys())
        self.make_data_frame()

        self.plot_colors = ["#0000ff", "#00aa00", "#aa0000", "#ff7700", "#aa00aa"]

    def load_data(self):
        data_by_user = dict()
        for path, dir, files in os.walk(self.data_directory):
            user_dir = path[(len(self.data_directory)+1):]
            if user_dir is not "":
                if len(files) > 0:
                    user_data = dict()
                    for file in files:
                        file_data = PickleUtil(os.path.join(path, file)).safe_load()
                        if "dist_id" in file:
                            user_data["click_dist"] = file_data
                            self.click_dist = file_data
                            continue
                        else:
                            data_value_names = {'errors', 'selections', 'characters', 'presses_sel', 'presses_char',
                                                   'presses_word', 'kde_mses', 'kde'}
                            self.param_names = set(file_data.keys()) - data_value_names
                            params = tuple(file_data[name] for name in self.param_names)
                        # print(self.param_names)
                        # print(params)
                        user_data[params] = file_data
                    data_by_user[int(user_dir)] = user_data
        return data_by_user

    def plot_across_user(self, metric, params=None, trends=False, log=False, legend=None):
        if isinstance(metric, str):
            metric = [metric]
        for m in metric:
            dep_vars = []
            ind_vars = []
            for user in self.user_numbers:
                user_data = self.data_by_user[user]

                if params is None:
                    params = list(user_data.keys())[1]
                else:
                    if params not in user_data.keys():
                        raise KeyError("Parameters are not in saved data")

                if m not in user_data[params].keys():
                    raise KeyError("Metric is not in saved data")

                dep_vars += [user_data[params][m]]
                if "attribute" in user_data[params]:
                    if "supercloud_results_20" in self.data_directory:
                        ind_vars += [user_data[params]["attribute"]*4]
                    else:
                        ind_vars += [user_data[params]["attribute"]]
                else:
                    ind_vars += [user]

            data_points = list(zip(ind_vars, dep_vars))
            data_points.sort()
            ind_vars, dep_vars = zip(*data_points)

            if np.array(dep_vars[0]).size == 1:
                if log:
                    dep_vars = np.log(dep_vars)

                plt.plot(ind_vars, dep_vars)

            else:
                colors = np.arange(len(ind_vars))
                colors = colors/(len(ind_vars))
                if trends:
                    avg_grads = []
                    for i, line in enumerate(dep_vars):
                        label = ind_vars[i]
                        x_values = np.arange(line.size)+1
                        line_norm = line-np.min(line)

                        smoothing = 20
                        smooth_x = x_values[smoothing:-smoothing]
                        smooth_line = np.convolve(line_norm, np.ones((smoothing,))/smoothing)[smoothing:len(x_values)-smoothing]
                        # plt.plot(smooth_x, np.gradient(smooth_line), color=(min(1, (1-colors[i])*2),0,min(1, colors[i]*2)))
                        avg_grads += [-np.average(np.gradient(smooth_line))]
                    plt.plot(ind_vars[1:], avg_grads[1:] / np.max(avg_grads))
                else:
                    plt.figure(figsize=(10, 12))
                    x_pos = np.log(max([s.size for s in dep_vars])*1.05)
                    plt.xlim(np.log(4), x_pos*1.1)
                    max_y = -float("inf")
                    for i, line in enumerate(dep_vars[1:]):
                        label = ind_vars[i]
                        x_values = np.arange(line.size) + 1

                        if log:
                            line = np.log(line)
                        x_values = np.log(x_values)

                        max_y = max(max_y, max(line))

                        plt.plot(x_values[5:], line[5:],
                                 color=(min(1, (1 - colors[i]) * 2), 0, min(1, colors[i] * 2)))

                        y_pos = line[-1] - 0.0005

                        plt.text(x_pos, y_pos, str(label), fontsize=12, color=(min(1, (1 - colors[i]) * 2), 0, min(1, colors[i] * 2)))
                    if legend is not None:
                        plt.text(x_pos/1.075, max_y+abs(max_y*0.0075), legend["multi"], fontsize=11)

        if legend is not None:
            plt.title(legend["title"])
            plt.xlabel(legend["x"])
            plt.ylabel(legend["y"])
        plt.show()

    def make_data_frame(self):
        average_data = {}
        num_users = len(self.user_numbers)
        for user in self.user_numbers:
            user_data = self.data_by_user[user]
            for param in user_data:
                if param != "click_dist":
                    if param not in average_data:
                        average_data[param] = {'errors': [], 'selections': [], 'characters': [], 'presses_sel': [],
                                               'presses_char': []}
                    for data_label in ['selections', 'characters', 'presses_sel', 'presses_char', 'errors']:
                        average_data[param][data_label] += user_data[param][data_label]

        data_labels = {'errors', 'selections', 'characters', 'presses_sel', 'presses_char'}
        param_name_dict = {'num_words': "Word Predictions Max Count", 'time_rotate': "Time Rotate",
                           'win_diff': "Win Difference", 'N_pred': "Words Per Character",
                           'prob_thresh': "Probability Threshold", 'attribute': 'Attribute',
                           'false_positive': 'False Positive Rate'}
        var_name_dict = {'selections': "Selections/Min", 'characters': "Characters/Min",
                         'presses_char': "Clicks/Character",
                         'presses_sel': "Clicks/Selection", 'errors': "Error Rate"}

        long_form_data = []
        for params in average_data.keys():
            param_names = [param_name_dict[name] for name in self.param_names]
            observation = dict(zip(param_names, params))

            param_data = average_data[params]
            num_observations = len(param_data['errors'])

            for obs in range(num_observations):
                for data_label in data_labels:
                    observation[var_name_dict[data_label]] = param_data[data_label][obs]

                long_form_data += [observation.copy()]

        df = pd.DataFrame(long_form_data)
        self.DF = df

        self.average_DF = []
        time_rotates = list(set(self.DF["Time Rotate"]))
        time_rotates.sort(key=lambda x: x)

        for tr in time_rotates:
            DF = df[df["Time Rotate"] == tr]
            self.average_DF += [np.average(DF, axis=0)]
        self.average_DF = pd.DataFrame(self.average_DF, columns=self.DF.columns)
        self.average_DF["Time Rotate"] = np.round(self.average_DF["Time Rotate"], 2)

        self.DF = self.DF[["Time Rotate", "Attribute", "Error Rate", "Selections/Min", "Clicks/Selection"]]

    def calc_stds(self, std_lookup):
        self.DF["STD"] = std_lookup[self.DF["Attribute"].values-1]

    def plot_across_params(self, params=None, sub_plot=None):

        self.DF['Selections/Min'] = np.array(self.DF['Selections/Min'].values, dtype=float)
        ind_var_name = "STD"

        if params is None:
            params = ['Error Rate', 'Selections/Min', 'Clicks/Selection',
                        'Error Rate']

        for dep_var_name in params:

            DF = self.DF[(self.DF["Time Rotate"] <= 10) & (self.DF["STD"] <= 1)]
            DF = DF[~DF.isin([np.nan, np.inf, -np.inf]).any(1)]

            DF = DF.groupby(["STD", "Time Rotate"]).mean()
            DF = DF.reset_index()
            pd.set_option('display.max_columns', 500)

            if sub_plot is None:
                fig, ax = plt.subplots()
            else:
                fig, ax = sub_plot

            fig.set_size_inches(10, 8)
            sns.set(font_scale=1.5, rc={"lines.linewidth": 3})
            sns.set_style({'font.serif': 'Helvetica'})

            # sns.lineplot(x=ind_var_name, y=dep_var_name, hue = "Time Rotate",
            #              palette=sns.cubehelix_palette(21, start=2, rot=0.2, dark=.2, light=.7, reverse=True),
            #              data=DF, ci="sd")

            ind_var_values = list(set(DF[ind_var_name].values))
            ind_var_values.sort()
            dep_var_values = list(set(DF["Time Rotate"].values))
            dep_var_values.sort()

            # im_array = np.zeros((len(ind_var_values), len(dep_var_values)))
            # for ind_index, ind_val in enumerate(ind_var_values):
            #     for dep_index, dep_val in enumerate(dep_var_values):
            #
            #         data = DF[(DF[ind_var_name] == ind_val) & ((DF["Time Rotate"] == dep_val))][dep_var_name].values
            #         if len(data) > 0:
            #             im_array[ind_index, dep_index] = np.mean(data)
            #         else:
            #             im_array[ind_index, dep_index] = np.NaN

            im_array = DF.pivot_table(index="STD", columns='Time Rotate', values=dep_var_name).values
            # im_array = np.where(np.isnan(im_array), 0, im_array)

            interp_points_raw = DF[["STD", "Time Rotate"]].values
            interp_points = []
            for point in interp_points_raw:
                interp_points.append([ind_var_values.index(point[0]), dep_var_values.index(point[1])])

            interp_values = DF[dep_var_name].values
            im_array_interpolator = NearestNDInterpolator(interp_points, interp_values)

            for point in np.array(np.where(np.isnan(im_array))).T:
                im_array[point[0], point[1]] = im_array_interpolator(point[0], point[1])

            norm_im_array = (im_array.T / np.max(im_array, axis=1)).T
            # fig, ax = plt.subplots()
            std_devs = np.array(ind_var_values)
            fill_detail = 0.1
            fill_levels = np.arange(0, 1+fill_detail, fill_detail)

            cont_detail = 2
            og_max = np.max(im_array)

            norm_im_array = gaussian_filter(norm_im_array, (1, 2))
            norm_im_array[0:, 0:] = gaussian_filter(norm_im_array[0:, 0:], (2, 10))
            im_array = gaussian_filter(im_array, (1, 2))
            im_array[5:, 0:] = gaussian_filter(im_array[5:, 0:], (2, 10))


            im_array /= np.max(im_array)/og_max

            def interpolate(x, x_vals, Z, max=True):
                diff_vals = x_vals - x
                lower_x = np.argmax(np.where(diff_vals < 0, diff_vals, -np.inf))
                higher_x = np.argmin(np.where(diff_vals > 0, diff_vals, np.inf))

                if max:
                    lower_z = Z[lower_x]
                    higher_z = Z[higher_x]
                else:
                    lower_z = lower_x
                    higher_z = higher_x

                return np.round((lower_z*(x_vals[higher_x]-x) + higher_z*(x-x_vals[lower_x]))/(x_vals[higher_x]-x_vals[lower_x]), 2)

            calc_level_val = np.vectorize(lambda x, y:  interpolate(x, std_devs, np.max(im_array, axis=1), max=y))
            std_levels = np.arange(0.25, 2, 0.25)
            cont_levels = calc_level_val(std_levels/2, True)
            cont_levels.sort()
            print(cont_levels)


            cmap_contourfill = sns.cubehelix_palette(len(fill_levels), start=2, rot=0.6, reverse=True, light=0.9, dark=0,
                                                     as_cmap=True)
            plt.contourf(dep_var_values, 2 * std_devs, gaussian_filter(norm_im_array, (0, 0)),
                         levels=fill_levels, cmap =cmap_contourfill)
            cbar = plt.colorbar()
            cbar.ax.set_xlabel('(%)')

            for i in std_levels:
                ax.axhline(i, color="w", linewidth=1, zorder=1)

            for i in np.arange(1, 10, 1):
                ax.axvline(i, color="w", linewidth=1, zorder=1)

            # cmap_contourline = sns.cubehelix_palette(len(cont_levels), start=0, rot=0.6, reverse=True, light=0.8,
            #                                          dark=0.4, as_cmap=True)
            # CS = plt.contour(dep_var_values, 2 * std_devs, im_array, levels=cont_levels, cmap = cmap_contourline, zorder=2)
            # ax.clabel(CS, inline=True, fontsize=12, fmt='%1.1f', manual=True, colors="#98443B")
            # cbar = plt.colorbar(ax=[ax], location='left', format='%1.1f')
            # cbar.ax.set_xlabel('(sel/min)')

            max_points_y = np.array(dep_var_values)[np.argmax(im_array, axis=1)][3:]
            plt.scatter(max_points_y, std_devs[3:]*2, s=100, facecolors='none', edgecolors='b', label="Horizontal Maximums", zorder=3)

            model = np.poly1d(np.polyfit(std_devs[3:-7]*2, max_points_y[:-7], 1))

            # add fitted polynomial line to scatterplot
            polyline = np.linspace(np.min(std_devs*2), np.max(std_devs*2), 50)
            plt.plot(model(polyline), polyline, label="RT = $13.00\sigma+0.584$")
            plt.legend(loc="upper left")

            # study_df = pd.read_csv("D:\\Users\\nickb\\PycharmProjectsD\\Nomon\\keyboard\\simulations\\increasing_variance\\study_click_dists.csv")
            # study_df = study_df[study_df["session"] == 10]
            # plt.scatter(study_df["time_rotate"], study_df["2_sigma"])

            ax.set_aspect(4)  # you may also use am.imshow(..., aspect="auto")
            plt.xlim([np.min(dep_var_values), np.max(dep_var_values)])
            plt.ylim([np.min(2 * std_devs), np.max(2 * std_devs)])
            params = {'mathtext.default': 'regular'}
            plt.rcParams.update(params)
            plt.xlabel('Rotation Time (s)')
            plt.ylabel('$2\sigma$ of Click Distribution (s)')
            plt.title("Selections/min across Rotation Times and Precision Levels")

            plt.show()

        return fig, ax


def order_data(dir):
    click_dists = []
    if not os.path.exists(os.path.join(dir, "ordered_data")):
        os.mkdir(os.path.join(dir, "ordered_data"))

    for path, __, files in os.walk(dir):
        for file in files:
            if "dist_id" in file:
                click_dist = PickleUtil(os.path.join(path, file)).safe_load()
                if click_dist not in click_dists:
                    click_dists += [click_dist]
                    # os.mkdir(os.path.join(dir, os.path.join("ordered_data", str(click_dists.index(click_dist)))))
                print(path)
                plt.plot(click_dist)
                plt.show()


def main():
    # rc_sdu = SimDataUtil_row_col("C:\\Users\\nickb\\PycharmProjects\\Nomon\\keyboard\\sim_data")
    # rc_sdu.plot_across_params(params=['Error Rate'])
    # plt.show()

    # param = 'Error Rate'
    # param = 'Clicks/Character'
    # sub_plot = rc_sdu.plot_across_params(params=[param])
    sdu_breadth = SimDataUtil("D:\\Users\\nickb\\PycharmProjectsD\\Nomon\\keyboard\\simulations\\increasing_variance\\supercloud_results")
    sdu_breadth.calc_stds(np.round(((2 - 0.03) / 20 * np.arange(1, 21) + 0.01), 3))
    sdu_min = SimDataUtil("D:\\Users\\nickb\\PycharmProjectsD\\Nomon\\keyboard\\simulations\\increasing_variance\\sim_data")
    sdu_min.calc_stds(np.round(((0.5-0.01)/20* np.arange(1, 21)+0.01), 3))
    sdu_breadth.DF = sdu_breadth.DF.append(sdu_min.DF)
    sdu_breadth.plot_across_params()


if __name__ == '__main__':
    main()
