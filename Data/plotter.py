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
from plot_helper import * 

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]
# -

# # Load Data and plot Bars

# +
# -1) Move any csv file into your Data directory 
# 0) Specify Data directory for the models
data_dir = "/raid/clark/summer2021/datasets/corrected-wander-full/data"

# 1) Specify the Dataset
df = pd.read_csv("wander_pre/wander_pre_percentage.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]
df = df.assign(clean_names=list(map(get_network_name, list(df["Network"]))))

# 2) Get extra Pretrained/Not Pretrained Data
avg_valid = merge_loss_data(data_dir, df, "valid_loss", "pretrained", True)
replicate_valid = merge_loss_data(data_dir, df, "valid_loss", "pretrained", False)
avg_train = merge_loss_data(data_dir, df, "train_loss", "pretrained", True)
replicate_train = merge_loss_data(data_dir, df, "train_loss", "pretrained", False)
avg_accuracy = merge_loss_data(data_dir, df, "accuracy", "pretrained", True)
df = pd.merge(df, avg_valid, on="clean_names")
df = df.sort_values(by=['mean_completion'])

# 3) Pass in "df" into the function, plot_bars()
ax = plot_bars(df, "Completion")
plt.gcf().subplots_adjust(left=0.35)
ax.set_title("Pretrained Wander Dataset Percentage Navigated Per Model Replicate");
# -

ax = plot_average_scatter(avg_accuracy[avg_accuracy.clean_names != "ResNet18"])
ax.title("Wander Dataset Pretrained Average Maze Percentage Navigated vs Validation Accuracy", size=20);
ax.xlabel("Validation Accuracy", size=20)
ax.ylabel("Average Percentage Navigated", size=20)
ax.xticks(size=15)
ax.yticks(size=15)

# The other DataFrames like "avg_valid" can be passed into "plot_average_scatter"()
ax = plot_average_scatter(avg_valid)
ax.title("Wander Dataset Pretrained Average Maze Percentage Navigated vs Validation Loss", size=20);
ax.xlabel("Validation Loss", size=20)
ax.ylabel("Average Percentage Navigated", size=20)
ax.xticks(size=15)
ax.yticks(size=15)

# +
df = pd.read_csv("wander_notpretrained/wander_npre_percentage.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]
df = df.assign(clean_names=list(map(get_network_name, list(df["Network"]))))
avg_valid = merge_loss_data(data_dir, df, "valid_loss", "notpretrained", True)
replicate_valid = merge_loss_data(data_dir, df, "valid_loss", "notpretrained", False)
avg_train = merge_loss_data(data_dir, df, "train_loss", "notpretrained", True)
replicate_train = merge_loss_data(data_dir, df, "train_loss", "notpretrained", False)
avg_accuracy = merge_loss_data(data_dir, df, "accuracy", "notpretrained", True)
df = pd.merge(df, avg_valid, on="clean_names")
df = df.sort_values(by=['mean_completion'])

# %load_ext autoreload
# %autoreload 2
ax = plot_bars(df, "Completion")
plt.gcf().subplots_adjust(left=0.35)
ax.set_title("Scratch Wander Dataset Percentage Navigated Per Model Replicate");
# -

ax = plot_average_scatter(avg_accuracy)
ax.title("Wander Dataset Scratch Average Maze Percentage Navigated vs Validation Accuracy", size=20);
ax.xlabel("Validation Accuracy", size=20)
ax.ylabel("Average Percentage Navigated", size=20)
ax.xticks(size=15)
ax.yticks(size=15)

ax = plot_average_scatter(avg_valid)
ax.title("Wander Dataset Scratch Average Maze Percentage Navigated vs Validation Loss", size=20);
ax.xlabel("Validation Loss", size=20)
ax.ylabel("Average Percentage Navigated", size=20)
ax.xticks(size=15)
ax.yticks(size=15)

pd.read_csv("wander_cmd/cmd_percentage.csv", index_col=0)

# +
data_dir = "/raid/clark/summer2021/datasets/corrected-wander-full/cmd_data"

df = pd.read_csv("wander_cmd/cmd_percentage.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]
df = df.assign(clean_names=list(map(get_network_name, list(df["Network"]))))
avg_valid = merge_loss_data(data_dir, df, "valid_loss", "cmd", True)
replicate_valid = merge_loss_data(data_dir, df, "valid_loss", "cmd", False)
avg_train = merge_loss_data(data_dir, df, "train_loss", "cmd", True)
replicate_train = merge_loss_data(data_dir, df, "train_loss", "cmd", False)
df = pd.merge(df, avg_valid, on="clean_names")
df = df.sort_values(by=['mean_completion'])

# %load_ext autoreload
# %autoreload 2
ax = plot_bars(df, "Completion")
plt.gcf().subplots_adjust(left=0.35)
ax.set_title("Not Pretrained Percentage Navigated Per Wander Dataset Model Replicate");

# +
df = pd.read_csv("RNN/rnn_percentage.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]
df = df.assign(clean_names=list(map(get_network_name, list(df["Network"]))))
avg_valid = merge_loss_data(data_dir, df, "valid_loss", "rnn", True)
replicate_valid = merge_loss_data(data_dir, df, "valid_loss", "rnn", False)
avg_train = merge_loss_data(data_dir, df, "train_loss", "rnn", True)
replicate_train = merge_loss_data(data_dir, df, "train_loss", "rnn", False)
df = pd.merge(df, avg_valid, on="clean_names")
df = df.sort_values(by=['mean_completion'])

# %load_ext autoreload
# %autoreload 2
ax = plot_bars(df, "Completion")
plt.gcf().subplots_adjust(left=0.35)
ax.set_title("Not Pretrained Percentage Navigated Per Wander Dataset Model Replicate");

# +
df = pd.read_csv("uniform_pretrained/uniform_pre_completion_results.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]
df = df.assign(clean_names=list(map(get_network_name, list(df["Network"]))))
avg_valid = merge_loss_data(data_dir, df, "valid_loss", "pretrained", True)
replicate_valid = merge_loss_data(data_dir, df, "valid_loss", "pretrained", False)
avg_train = merge_loss_data(data_dir, df, "train_loss", "pretrained", True)
replicate_train = merge_loss_data(data_dir, df, "train_loss", "pretrained", False)
df = pd.merge(df, avg_valid, on="clean_names")
df = df.sort_values(by=['mean_completion'])

# %load_ext autoreload
# %autoreload 2
ax = plot_bars(df, "Completion")
plt.gcf().subplots_adjust(left=0.35)
ax.set_title("Pretrained Uniform Dataset Percentage Navigated Per Model Replicate");

# +
df = pd.read_csv("uniform_nonpre/uniform_nonpre_percentage.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]
df = df.assign(clean_names=list(map(get_network_name, list(df["Network"]))))
avg_valid = merge_loss_data(data_dir, df, "valid_loss", "notpretrained", True)
replicate_valid = merge_loss_data(data_dir, df, "valid_loss", "notpretrained", False)
avg_train = merge_loss_data(data_dir, df, "train_loss", "pretrained", True)
replicate_train = merge_loss_data(data_dir, df, "train_loss", "notpretrained", False)
df = pd.merge(df, avg_valid, on="clean_names")
df = df.sort_values(by=['mean_completion'])

# %load_ext autoreload
# %autoreload 2
ax = plot_bars(df, "Completion")
plt.gcf().subplots_adjust(left=0.35)
ax.set_title("Scratch Uniform Dataset Percentage Navigated Per Model Replicate");

# +
df = pd.read_csv("handmade_pretrained/handmade_pre_percentage.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]
df = df.assign(clean_names=list(map(get_network_name, list(df["Network"]))))
avg_valid = merge_loss_data(data_dir, df, "valid_loss", "pretrained", True)
replicate_valid = merge_loss_data(data_dir, df, "valid_loss", "pretrained", False)
avg_train = merge_loss_data(data_dir, df, "train_loss", "pretrained", True)
replicate_train = merge_loss_data(data_dir, df, "train_loss", "pretrained", False)
df = pd.merge(df, avg_valid, on="clean_names")
df = df.sort_values(by=['mean_completion'])

# %load_ext autoreload
# %autoreload 2
ax = plot_bars(df, "Completion")
plt.gcf().subplots_adjust(left=0.35)
ax.set_title("Pretrained Handmade Dataset Percentage Navigated Per Model Replicate");

# +
df = pd.read_csv("handmade_notpretrained/handmade_notpre_percentage.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]
df = df.assign(clean_names=list(map(get_network_name, list(df["Network"]))))
avg_valid = merge_loss_data(data_dir, df, "valid_loss", "notpretrained", True)
replicate_valid = merge_loss_data(data_dir, df, "valid_loss", "notpretrained", False)
avg_train = merge_loss_data(data_dir, df, "train_loss", "notpretrained", True)
replicate_train = merge_loss_data(data_dir, df, "train_loss", "notpretrained", False)
df = pd.merge(df, avg_valid, on="clean_names")
df = df.sort_values(by=['mean_completion'])

# %load_ext autoreload
# %autoreload 2
ax = plot_bars(df, "Completion")
plt.gcf().subplots_adjust(left=0.35)
ax.set_title("Scratch Handmade Dataset Percentage Navigated Per Model Replicate");
# -



# +
# %load_ext autoreload
# %autoreload 2

ax = plot_average_scatter(avg_accuracy)
ax.title("Wander Dataset Scratch Average Maze Percentage Navigated vs Validation Accuracy", size=20);
ax.xlabel("Validation Accuracy", size=20)
ax.ylabel("Average Percentage Navigated", size=20)
ax.xticks(size=15)
ax.yticks(size=15)
# -

ax2 = plot_average_scatter(replicate_valid)
ax.title("Maze Percentage Navigated as a Function of Valid Loss", size=20);
ax.xlabel("Valid Loss", size=20)
ax.ylabel("Percentage Navigated", size=20)
ax.xticks(size=15)
ax.yticks(size=15)

ax3 = plot_average_scatter(avg_train)
ax.title("Average Maze Percentage Navigated as a Function of Training Loss", size=20);
ax.xlabel("Training Loss", size=20)
ax.ylabel("Average Percentage Navigated", size=20)
ax.xticks(size=15)
ax.yticks(size=15)

ax4 = plot_average_scatter(replicate_train)
ax.title("Maze Percentage Navigated as a Function of Training Loss", size=20);
ax.xlabel("Training Loss", size=20)
ax.ylabel("Percentage Navigated", size=20)
ax.xticks(size=15)
ax.yticks(size=15)

# %load_ext autoreload
# %autoreload 2
warnings.filterwarnings('ignore')
ax = plot_boxplot(df, mazes);
ax.title("Percentage Completed Per Maze", size=20)
ax.xlabel("Validation Maze", size=20)
ax.ylabel("Percentage Completed", size=20)
ax.xticks(size=10)
ax.yticks(size=10)

df = pd.read_csv("../Imitator/RNN_percentage_uniform_full_2_mazes.csv", index_col=0)
columns = list(df.columns)
mazes = columns[1:-3]

ax = plot_bars(df, "Completion")

# # Scraps 

model_type = []
for m in list(df['Network']):
    if "notpretrained" in m:
        model_type.append("Non-Pretrained")
    else:
        model_type.append("Pretrained")

clean_names = list(map(get_network_name, list(df['Network'])))

df = df.assign(model_type = model_type, clean_names = clean_names)
df

# # Scatter Plot Code

clean_names = list(map(get_network_name, list(pretrained_df['Network'])))

pretrained_df = pretrained_df.assign(clean_names = clean_names)

data_dir = "/raid/clark/summer2021/datasets/wander-full/data"

# Import seaborn
import seaborn as sns
# Apply the default theme
sns.set_theme()

fig = plt.gcf()
fig.set_size_inches(12, 9)
sns.scatterplot(data=pretrained_df, x="mean", y="error", hue="Network", style="Network", s=200)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

len(list(set(pretrained_df['clean_names'])))

colors = get_colors(24)

# +
sns_colors = list(sns.color_palette())

color_labels = []
for i in range(24 + 1):
    color = sns_colors[i % len(sns_colors)]
    for i in range(4):
        color_labels.append(color)
color_labels;

# +
fig = plt.gcf()
fig.set_size_inches(16, 9)
sns.set_theme(style="whitegrid")

ci_bounds = get_error(pretrained_df)

label_names = list(pretrained_df["clean_names"])
sparse_labels = []
for i in range(0, len(label_names)):
    if i % 4 == 0:
        sparse_labels.append(clean_names[i])
    else:
        sparse_labels.append("")

ax = sns.barplot(
    x="Network",
    y="mean",
    data=pretrained_df,
    palette=color_labels,
    yerr=ci_bounds,
    ecolor="grey",
#     saturation=1
)
ax.set_xticklabels(labels=sparse_labels, rotation=75)
ax.set_yticks(np.arange(0, 100, 10))
ax.set_ylim(bottom=0.0, top=100.0)
# -

avg_df = merge_loss_data(data_dir, pretrained_df, "valid_loss", clean_names, True)
ax = plot_average_scatter(avg_df)
ax.set_title("Averaged Steps Taken Over Valid Loss Per Model")
ax.set_xlabel("Valid Loss")

training_losses = []
for c in csvs:
    df = pandas.read_csv(data_dir + "/" + c)
    if len(df) == 0:
        training_losses.append(0.0)
    else:
        training_losses.append(min(df["valid_loss"]))
training_losses

losses

# +
data_files = os.listdir(data_dir)
data_files.sort()
csvs = list(filter(get_csv, data_files))
losses = get_losses(csvs, data_dir, "valid_loss")
means = pretrained_df['mean']
names = clean_names    
   
# losses = np.array(losses).reshape(-1,4).mean(axis=1)
# means = np.array(means).reshape(-1,4).mean(axis=1)
# names = list(set(clean_names))

# df = pandas.DataFrame()
# df = df.assign(Network = names)
# df = df.assign(losses = losses)
# df = df.assign(mean_steps = means)
# -

np.array(losses).reshape(-1,4).mean(axis=1)
