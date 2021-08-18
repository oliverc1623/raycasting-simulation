# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import matplotlib as plt
import sys
sys.path.insert(0, '../Imitator')
# from plot_helper import * 
data_dir = "/raid/clark/summer2021/datasets/handmade-full/data"

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]

# +
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
from pathlib import Path
import sys
import Imitate
from Imitate import *
import RegressionImitator
from RegressionImitator import *
import pandas
import seaborn as sns
from time import strftime
from time import gmtime

colors = [
    "tab:blue",
    "tab:orange",
    "tab:purple",
    "tab:red",
    "tab:green",
    "tab:brown",
    "tab:pink",
]

def get_network_name(m):
    name = m.split("-")[1]
    return (
        name.capitalize()
        .replace("net", "Net")
        .replace("resnext", "ResNext")
        .replace("deep", "Deep")
        .replace("deeper", "Deeper")
        .replace("_res", "_Res")
    )

def get_clean_names(m):
    return m.split("-")[1]

def index_one(num):
    return num[0]

def get_df(data_dict, mazes):
    df = pd.DataFrame.from_dict(data_dict)
    df = df.assign(Maze=mazes)
    df = df.T
    df.columns = df.iloc[-1]
    df = df.drop(df.index[-1])
    # df['Network'] = df.index
    df = df.reset_index()
    df.rename(columns={'index':'Network'}, inplace=True)
    df['std'] = df[df.columns[1:]].std(axis=1)
    df['mean'] = df[df.columns[1:-1]].mean(axis=1)
    # df = df.assign(lower_error=1.96*df['std'])
    df = df.assign(error=1.96*df['std']/np.sqrt(len(mazes)))
    return df

def get_error(df):
    ci_bounds = df['error'].to_numpy()
    return ci_bounds

def get_colors(num_unique_mazes) -> list:
    color_labels = []
    for i in range(num_unique_mazes):
        color = colors[i % len(colors)]
        for i in range(4):
            color_labels.append(color)
    return color_labels

def plot_bars(df, metric):
    matplotlib.rcParams.update({'font.size': 12})
    
    full_names = list(df["Network"])
    clean_names = list(map(get_network_name, full_names))
    unique_names = list(pd.unique(clean_names))
    sparse_labels = []

    for i in range(0, len(clean_names)):
        if i % 4 == 0:
            sparse_labels.append(clean_names[i])
        else: 
            sparse_labels.append("")
    
    color_labels = get_colors(len(unique_names))
    ci_bounds = get_error(df)
    max_error = max(df["error"])
    increment = 5
    fig, ax = plt.subplots(figsize=(12, 12))
    y_lab = "Average Steps Needed Over Averaged Mazes"

    x = np.arange(len(clean_names))
    width = 0.65
    vals = list(df["mean"])
    
    if metric == "Completion":
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        y_lab = "Average Completion Over Averaged Mazes"
        ax.set_xticks(np.arange(0, 100, 10))
        ax.set_xlim(left = 0.0, right = 100.0)
    else:
        ax.set_xticks(np.arange(0, max(vals), 200))
        ax.set_xlim(left = 0.0, right = max(vals))
    
    ax.barh(
        x,
        vals,
        xerr=ci_bounds,
        color=color_labels,
        align="center",
        alpha=0.75,
        ecolor="grey",
        capsize=2,
    )
    ax.set_yticks(x)
#     ax.set_yticks(np.arange(0, max(vals) + max_error, increment))
    ax.set_yticklabels(labels=sparse_labels)
    # ax.legend()
    # Axis styling.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#EEEEEE")
    ax.yaxis.grid(False)

    ax.set_xlabel("Percentage Completed", labelpad=15)
    ax.set_ylabel("Architecture", labelpad=15)
    ax.set_title(f"{metric} Navigated per Network Replicate")
    return ax 

# Scatter Plot Code
def get_csv(file):
    return "csv" in file

def get_petrained(file):
    return "-pretrained" in file

def get_nonpetrained(file):
    return "-notpretrained" in file

def min_to_sec(t):
    return int(t.split(':')[0]) * 60 + int(t.split(':')[1])

def get_losses(csv_files, data_dir, loss_type):
    training_losses = []
    for c in csv_files:
        if c == "classification-resnet18-pretrained-trainlog-0.csv":
            training_losses.append(0.0)
            continue
        df = pandas.read_csv(data_dir + "/" + c)
        if len(df) == 0:
            training_losses.append(0.0)
        else:
            if loss_type == "time":
                training_losses.append(min_to_sec(min(df[loss_type])))
            else:
                training_losses.append(min(df[loss_type]))
    return training_losses

def merge_loss_data(data_dir, df, loss_type, model_type, average=False):
    data_files = os.listdir(data_dir)
    data_files.sort()
    csvs = list(filter(get_csv, data_files))
    if model_type == "pretrained":
        csvs = list(filter(get_petrained, csvs))
    else: 
        csvs = list(filter(get_nonpetrained, csvs))
    losses = get_losses(csvs, data_dir, loss_type)
    means = df['mean']
    names = list(map(get_network_name, df['Network']))    
    
    if average:        
        losses = np.array(losses).reshape(-1,4).mean(axis=1)
        means = np.array(means).reshape(-1,4).mean(axis=1)
        names = list(pd.unique(names))
    
    df = pandas.DataFrame()
    df = df.assign(clean_names = names)
    df = df.assign(losses = losses)
    df = df.assign(mean_completion = means)
    return df

def plot_average_scatter(df):   
    matplotlib.rcParams.update({'font.size': 12})
    sns.set_theme()
#     sns.set_context("paper")
    fig = plt.gcf()
    fig.set_size_inches(12, 9)
    sns.scatterplot(data=df, x="losses", y="mean_completion", hue="clean_names", style="clean_names", s=200)
    plt.legend(bbox_to_anchor=(1.10, 1), borderaxespad=0)
    x = list(df['losses'])
    y = list(df['mean_completion'])
    for i, label in enumerate(list(df['clean_names'])):
        plt.annotate(label, (x[i], y[i]))
    return plt

def clean_maze_name(m):
    return m[:-4].capitalize().replace("_", " ")

def plot_boxplot(df, mazes):
    all_data = []
    for m in mazes:    
        x = list(df[m])
        all_data.append(x)
    all_data = pd.DataFrame(all_data).T
    all_data.columns = mazes
    all_data = pd.DataFrame(all_data.stack())
    all_data.reset_index(level=1, inplace=True)
    all_data.columns = ['maze', 'percentage']
    clean_mazes = list(map(clean_maze_name, list(all_data['maze'])))
    all_data = all_data.assign(maze=clean_mazes)
        
    sns.set_theme()
    sns.set_context("paper")
    fig = plt.gcf()
    fig.set_size_inches(17, 9)
    sns.swarmplot(x="maze", y="percentage", data=all_data)
    plt.ylim=(0, None)
    return plt

def convert_sec_to_min(s):
    return strftime("%M:%S", gmtime(s))


# -

# # Get Pretrained Data

# ## Get Percent, losses, and names

# +
df = pd.read_csv("handmade_pretrained/handmade_pre_percentage.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]
df = df.assign(clean_names=list(map(get_network_name, list(df["Network"]))))
avg_valid = merge_loss_data(data_dir, df, "valid_loss", "pretrained", True)
avg_train = merge_loss_data(data_dir, df, "train_loss", "pretrained", True)
df = pd.merge(df, avg_valid, on="clean_names")
df = df.sort_values(by=['mean_completion'])

summary_table = avg_valid.rename(columns={'losses':'Valid Loss'})
avg_train = avg_train.rename(columns={'losses':'Train Loss'})
summary_table = summary_table.assign(Train_Loss=avg_train["Train Loss"])
summary_table = summary_table[
    ["clean_names", "Valid Loss", "Train_Loss", "mean_completion"]
]
summary_table = summary_table.rename(
    columns={"Train_Loss": "Train Loss", "mean_completion": "Percent Navigated", "clean_names":"Model"}
)
# -

# ## Get Parameters

# +
model_dir = '/raid/clark/summer2021/datasets/handmade-full/models'
models = os.listdir(model_dir)
pre = []
nonpre = []
for m in models:
    if "-notpre" in m:
        nonpre.append(m)
    else:
        pre.append(m)
pre.sort()
nonpre.sort()
pre = pre[::4]
nonpre = nonpre[::4]

pre_parameters = []
nonpre_parameters = []

for p in pre:
    model = load_learner(model_dir + "/" + p)
    pre_parameters.append(sum(p.numel() for p in model.parameters()))
    
for p in nonpre:
    model = load_learner(model_dir + "/" + p)
    nonpre_parameters.append(sum(p.numel() for p in model.parameters()))
    
summary_table["Number of Parameters"] = pre_parameters
summary_table = summary_table.set_index(['Model', 'Number of Parameters'])
# -

summary_table.columns = pd.MultiIndex.from_product(
    [["Pretrained"], ["Valid Loss", "Train Loss", "Percent Navigated"]], names=["Trained/Pretrained:", "Data:"]
)

summary_table

# # NotPretrained

# +
npdf = pd.read_csv("uniform_nonpre/uniform_nonpre_percentage.csv", index_col=0)
columns = list(npdf.columns)
mazes = columns[1:-3]
npdf = npdf.assign(clean_names=list(map(get_network_name, list(npdf["Network"]))))
avg_valid = merge_loss_data(data_dir, npdf, "valid_loss", "-notpretrained", True)
avg_train = merge_loss_data(data_dir, npdf, "train_loss", "-notpretrained", True)
npdf = pd.merge(npdf, avg_valid, on="clean_names")
npdf = npdf.sort_values(by=['mean_completion'])

npsummary_table = avg_valid.rename(columns={'losses':'Valid Loss'})
avg_train = avg_train.rename(columns={'losses':'Train Loss'})
npsummary_table = npsummary_table.assign(Train_Loss=avg_train["Train Loss"])
npsummary_table = npsummary_table[
    ["clean_names", "Valid Loss", "Train_Loss", "mean_completion"]
]
npsummary_table = npsummary_table.rename(
    columns={"Train_Loss": "Train Loss", "mean_completion": "Percent Navigated", "clean_names":"Model"}
)

# +
npsummary_table["Number of Parameters"] = pre_parameters
npsummary_table = npsummary_table.set_index(['Model', 'Number of Parameters'])

npsummary_table.columns = pd.MultiIndex.from_product(
    [["Not Pretrained"], ["Valid Loss", "Train Loss", "Percent Navigated"]], names=["Trained/Pretrained:", "Data:"]
)
# -

d = {'Pretrained' : summary_table['Pretrained'], 'Not Pretrained' : npsummary_table['Not Pretrained']} 
combined_data = pd.concat(d.values(), axis=1, keys=d.keys())

combined_data.columns = combined_data.columns.rename("Pretrained/Not Pretrained", level=0)

# # Get Time

df=df.sort_values(by="Network")
avg_time = merge_loss_data(data_dir, df, "time", "pretrained", True)
avg_time['losses'] = list(map(convert_sec_to_min, list(avg_time['losses'])))
avg_time.columns = ['clean_names', 'time', 'mean_completion']

avg_time_np = merge_loss_data(data_dir, df, "time", "-notpretrained", True)
avg_time_np['losses'] = list(map(convert_sec_to_min, list(avg_time_np['losses'])))
avg_time_np.columns = ['clean_names', 'time', 'mean_completion']

combined_data['Pretrained', 'Time'] = list(avg_time['time'])
combined_data['Not Pretrained', 'Time'] = list(avg_time_np['time'])

combined_data = combined_data.sort_index(axis=1)

combined_data

# +
s = combined_data.style

cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#ffffb3')]
}
index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
}
headers = {
    'selector': '.col_heading',
    'props': 'background-color: #000066; color: white;'
}
row_headers = {
    'selector': '.row_heading',
    'props': 'background-color: #000066; color: white;'

}
s.set_table_styles([cell_hover, index_names, headers, row_headers])

s.set_table_styles([
    {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5em;'},
    {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'},
], overwrite=False)

s.set_table_styles({
    ('Pretrained', 'Percent Navigated'): [{'selector': 'th', 'props': 'border-left: 1px solid white'},
                               {'selector': 'td', 'props': 'border-left: 1px solid #000066'}]
}, overwrite=False, axis=0)

cm = sns.light_palette("green", as_cmap=True)

s.background_gradient(cmap=cm)
# -

s.to_latex("table")

pre_nav = list(combined_data['Not Pretrained', 'Percent Navigated'])
npre_nav = list(combined_data['Pretrained', 'Percent Navigated'])

import scipy

# # Apply two-sample T-Test

# $H_0: \mu_{Pretrained} = \mu_{Not Pretrained}$
#
# $H_A: \mu_{Pretrained} \ne \mu_{Not Pretrained}$

scipy.stats.ttest_ind(pre_nav, npre_nav)

nav_df = pd.DataFrame([pre_nav, npre_nav]).T
nav_df.columns = ['non_pretrained', 'pretrained']

# Based on the p-value, 0.0688, we fail to reject the null hypothesis that the population means of the pretrained and non-pretrained groups are equal.

# +
pre_nav = list(combined_data['Not Pretrained', 'Percent Navigated'])
npre_nav = list(combined_data['Pretrained', 'Percent Navigated'])

nav_df = pd.DataFrame([pre_nav, npre_nav]).T
nav_df.columns = ['non_pretrained', 'pretrained']

sns.kdeplot(data=nav_df)
# -

model_dir = '/raid/clark/summer2021/datasets/corrected-wander-full/models'
models = os.listdir(model_dir)
models = list(filter(lambda x: "-notpretrained" in x, models))
models.sort()

models

models[16:20]

m = load_learner(model_dir + "/classification-resnet50-pretrained-0.pkl")

m = torch.load(model_dir + "/classification-resnet50-pretrained-0.pkl", "cuda:3")

# +
import matplotlib.pyplot as plt
import sys
sys.path.append("../PycastWorld")
sys.path.append("../Gym")
from gym_pycastworld.PycastWorldEnv import PycastWorldEnv

sys.path.append("../Automator")
from AutoGen import Navigator

from fastai.vision.all import *
import datetime;

import sys
import fastprogress
from matplotlib.animation import FuncAnimation

from IPython.display import HTML
# sys.path.append("../PycastWorld")
# from gym_pycastworld.PycastWorldEnv import PycastWorldEnv  # type: ignore
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(3)
else:
    device = torch.device('cpu')

# +
steps_per_episode = 1500

env = PycastWorldEnv("../Mazes/maze01.txt", 224, 224)

observation = env.reset()
frames = [observation.copy()]
model_inf = load_learner(model_dir + "/classification-resnet50-pretrained-0.pkl", cpu=False)
# model_inf = torch.load(model_dir + "/classification-resnet50-pretrained-0.pkl", "cuda:3")
model_inf.eval();
# -

model_inf.to("cuda:3");

model_inf.predict(observation)


