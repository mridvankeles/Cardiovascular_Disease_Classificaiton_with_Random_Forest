import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
import configparser

def visualize_box_plots(df,df_new,name="box_plot"):
    cols = ["age","height","weight","ap_hi","ap_lo"]
    f,axes = plt.subplots(2,5,figsize=[30,15])
    for i,t in enumerate(cols):
        boxplot = sns.boxplot(x="cardio",y =t,data=df,orient = "v",ax=axes[0,i%5])
        boxplot = sns.boxplot(x="cardio",y =t,data=df_new,orient = "v",ax=axes[1,i%5])

    plt.title(f"{name}box plot raw/processed features")
    boxplot.figure.savefig(f"reports/figures/boxplots/{name}.png")
    plt.close()


def visualize_cat_plots(df,name="cat_plot"):
    cols = ["gender","cholesterol","gluc","smoke","alco","active","cardio"]
    f,axes = plt.subplots(ncols=7,figsize=[35,8])
    for i,t in enumerate(cols):
        catplot = sns.countplot(data=df, x=t, hue="cardio",orient = "h",ax=axes[i%7])
        plt.title(f"{t}catplot processed feature")

    plt.title(f"{name}categorical plot processed features")
    catplot.figure.savefig(f"reports/figures/catplots/{name}.png")
    plt.close()

def visualize_age(df):
    sns_plot = sns.countplot(x='age', hue='cardio', data=df, palette="Set2")
    plt.title('Effect of age feature to cardio target')
    plt.xlabel("AGE")
    plt.ylabel("CARDIO")
    sns_plot.figure.savefig("reports/figures/countplots/visualize_age.png")
def visualize_alch(df):
    sns_plot = sns.catplot(data=df, kind="count", x="alco", hue="cardio")
    plt.title('Effect of alco feature to cardio target')
    plt.xlabel("ALCO")
    plt.ylabel("CARDIO")
    sns_plot.figure.savefig("reports/figures/catplots/visualize_alcohol.png")
def visualize_smoke(df):
    sns_plot = sns.catplot(data=df, kind="count", x="smoke", hue="cardio")
    plt.title('Effect of smoke feature to cardio target')
    plt.xlabel("SMOKE")
    plt.ylabel("CARDIO")
    sns_plot.figure.savefig("reports/figures/catplots/visualize_smoke.png")
def visualize_cats(df):
    df_long = pd.melt(df, id_vars=['cardio'], value_vars=[
                      'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
    sns_plot = sns.catplot(x="variable", hue="value",
                           col="cardio", data=df_long, kind="count")
    plt.title('effects of categorical features to target')
    plt.xlabel("categorical features")
    plt.ylabel("counts")
    sns_plot.figure.savefig("reports/figures/catplots/visualize_cats.png")
def visualize_cholesterol(df):
    sns_plot = sns.catplot(data=df, kind="count", x="cholesterol", hue="cardio")
    plt.title('effects of cholesterol feature to target')
    plt.xlabel("CHOLESTEROL")
    plt.ylabel("CARDIO")
    sns_plot.figure.savefig("reports/figures/catplots/visualize_cholesterol.png")
def visualize_cholesterol_and_gluc(df):
    sns_plot = sns.catplot(data=df, kind="bar",
                           x="cholesterol", y="gluc", hue="cardio")
    plt.title('cholesterol and gluc features visualization')
    sns_plot.figure.savefig(
        "reports/figures/catplots/visualize_cholesterol_and_gluc.png")


def visualize_features(df,features):
    plt.figure(figsize = (25,15))
    plt.title('displot plot of all features to target')
    for feature in tqdm(features):
        sns_plot=sns.displot(df[feature])
        sns_plot.figure.savefig(r'reports/figures/dist_pair_scatter_plots/'+feature+'_'+"dist"+'.png')
        plt.close()

    plt.figure(figsize = (35,25))
    plt.title('scatter plot of all features to target')
    sns_plot=sns.pairplot(data=df,x_vars=features[:-2],y_vars=features[-2:])
    sns_plot.figure.savefig(r'reports/figures/dist_pair_scatter_plots/'+"all_features"+'_'+"pairplot"+'.png')
    plt.close()


def correlation_matrix(df,path):
    corr = df.corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 15))
    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
                           square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('triangle correlation matrix')
    sns_plot.figure.savefig(path)


def other_correlation_matrix(df,path):
    plt.rcParams['figure.figsize'] = (25, 15)
    sns_plot = sns.heatmap(round(df.corr(),3), annot=True, linewidths=.5, cmap="YlGnBu")
    plt.title('correlation matrix')
    sns_plot.figure.savefig(path)

def joint_plot_for_numerics(df,name="jointplot"):
    cols = ["ap_hi","ap_lo","height","weight"]
    for i,t in enumerate(cols):
        catplot = sns.jointplot(data=df, x=t,y="ap_hi", hue="cardio")
        catplot.figure.savefig(f"reports/figures/jointplots/{name}ap_hi{t}.png")
        plt.close()
        catplot = sns.jointplot(data=df, x=t,y="ap_lo", hue="cardio")
        catplot.figure.savefig(f"reports/figures/jointplots/{name}ap_lo{t}.png")
        plt.close()
        catplot = sns.jointplot(data=df, x=t,y="height", hue="cardio")
        catplot.figure.savefig(f"reports/figures/jointplots/{name}height{t}.png")
        plt.close()

def main():
    # get paths from config
    config = configparser.ConfigParser()
    config.read("config.ini")
    processed_paths = config["PROCESSED"]
    data_paths = config["DATASET"]

    np.random.seed(42)
    df = pd.read_csv(processed_paths["PROCESSED_TRAIN_PATH"], sep=";")
    raw_df = pd.read_csv(data_paths["TRAIN_PATH"], sep=";")
    new_df = pd.read_csv(processed_paths["ALL_FEATURES_TRAIN_PATH"], sep=";")

    visualize_box_plots(raw_df,new_df,"features boxplots before and after outlier procesess")
    visualize_features(df,df.columns[1:])
    visualize_cat_plots(raw_df,"count_plot")
    """  visualize_age(df)
    visualize_cats(df)
    visualize_cholesterol(df)
    visualize_alch(df)
    visualize_smoke(df)
    visualize_cholesterol_and_gluc(df)"""
    correlation_matrix(raw_df,path="reports/figures/corr_matrix/raw_triangle_correlation_matrix.png")
    other_correlation_matrix(raw_df,path="reports/figures/corr_matrix/raw_correlation_matrix.png")
    correlation_matrix(new_df,path="reports/figures/corr_matrix/new_triangle_correlation_matrix.png")
    other_correlation_matrix(new_df,path="reports/figures/corr_matrix/new_correlation_matrix.png")
    joint_plot_for_numerics(new_df,"jointplots")


if __name__ == "__main__":
    main()
