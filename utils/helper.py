import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

import os
import sys

from .constants import PATH_DATA



def restart_kernel():
    os.execl(sys.executable, sys.executable, *sys.argv)



def read_dataset(dataset_name):
    try:
        datasets_dict = {}
        cur_root_dir = PATH_DATA
        root_dir_dataset = cur_root_dir + '/' + dataset_name + '/'

        df_train = pd.read_csv(root_dir_dataset + dataset_name +
                            '_TRAIN.tsv', sep='\t', header=None)
        df_test = pd.read_csv(root_dir_dataset + dataset_name +
                            '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]

        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])

        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                    y_test.copy())

        return datasets_dict[dataset_name]
    except:
        print("Error fetching data")
        return [None]

def plot(dataset, labels):
    try:
        dataset_df = pd.DataFrame(dataset)
        labels_df = pd.DataFrame(labels, columns=['Label'])
        sampleLength = dataset_df.shape[1]
        data_for_each_label = []
        maxClassCount = labels_df.value_counts().max()

        for label in labels_df['Label'].unique():
            classData = dataset_df[labels_df['Label'] == label].values
            classCount = len(classData)
            if(classCount<maxClassCount):
                padding = np.zeros(((maxClassCount-classCount), sampleLength))
                classData = np.concatenate((classData, padding))
            data_for_each_label.append(classData)


        num_rows, num_cols = 4, 5
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))
        axes = axes.flatten()
        for j in range(num_rows * num_cols):
            if j < len(dataset_df.columns):
                column_name = dataset_df.columns[j]
                for df in data_for_each_label:
                    data = pd.DataFrame(df).T
                    if data[column_name].any():
                        axes[j].plot(data.index, data[column_name])
                axes[j].set_title(f'Time Series {j+1} of {len(data_for_each_label[0])}')

        plt.tight_layout()
        plt.show()
    except:
        print("Error")


def plot_pie_chart(original_labels, predicted_labels, title):
    original_counts = pd.DataFrame(original_labels).value_counts()
    predicted_counts = pd.DataFrame(predicted_labels).value_counts()
    labelsTrain = []
    for i in set(original_labels):
        labelsTrain.append(f"Class {i}")
    labelsTest = []
    for i in set(predicted_labels):
        labelsTest.append(f"Class {i}")
    # Plotting the pies
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))


    # Original Data Pie
    ax[0].pie(original_counts, labels=labelsTrain, autopct='%1.1f%%', startangle=90)
    ax[0].set_title('Original Data Classes')

    # Predicted Data Pie
    ax[1].pie(predicted_counts, labels=labelsTest, autopct='%1.1f%%', startangle=90)
    ax[1].set_title('Predicted Data Classes')

    fig.suptitle(title, fontsize=20)
    # Display the plot
    plt.show()

def plot_1v1_perf(res_df,column1,column2, acc_base=100):
    # Define the points for the diagonal line
    x_line = [0, acc_base]
    y_line = [0, acc_base]

    # Define points for scatter plot
    x_scatter = res_df[column1].tolist() 
    y_scatter = res_df[column2].tolist() 

    x_above = np.array([x for x, y in zip(x_scatter, y_scatter) if y > x])
    y_above = np.array([y for x, y in zip(x_scatter, y_scatter) if y > x])

    x_same = np.array([x for x, y in zip(x_scatter, y_scatter) if y == x])
    y_same = np.array([y for x, y in zip(x_scatter, y_scatter) if y == x])

    x_below = np.array([x for x, y in zip(x_scatter, y_scatter) if y < x])
    y_below = np.array([y for x, y in zip(x_scatter, y_scatter) if y < x])

    # Plot the diagonal line
    plt.plot(x_line, y_line,  color='blue')
    num_wins = res_df[res_df[column2] > res_df[column1]].shape[0]
    num_ties = res_df[res_df[column2] == res_df[column1]].shape[0]
    num_losses = res_df[res_df[column2] < res_df[column1]].shape[0]

    # Plot the scatter points
    plt.scatter(x_above, y_above, label='LogisticRegression Wins - ' + str(num_wins), color='red')
    plt.scatter(x_same, y_same, label='Equal - ' + str(num_ties), color='orange')
    plt.scatter(x_below, y_below, label=f'{column1} Wins - ' + str(num_losses), color='green')

    # # Set axis limits
    plt.xlim(0, acc_base)
    plt.ylim(0, acc_base)

    # Add labels and title
    plt.xlabel(f'{column1} perf.')
    plt.ylabel('LogisticRegression perf.')
    plt.title(f'1 x 1 Performance Comparison - {column1} and "LogisticRegression" ')

    # Add a legend
    plt.legend()
    plt.savefig("compare_results/"+ column1 + '_runs_10.png')
    plt.close()
    # Display the plot
    # plt.show()

def label_encoder(y):
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_map[label] for label in y])
    return encoded_labels


class Log:
    def __init__(self) -> None:
        pass

    def error(self,input):
        print(f"\033[91m {input} \033[00m")
    
    def success(self,input):
        print(f"\033[92m {input} \033[00m")

    def info(self,input):
        print(f"{input}")


def reset_gpu():
    # Clear GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.backend.clear_session()  # Clear TensorFlow session
            print("GPU memory has been successfully cleared.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available. Nothing to clear.")
    
    # Reset cuDNN, cuFFT, and cuBLAS factories
    try:
        tf.keras.backend.clear_session()  # Clear TensorFlow session again
        print("cuDNN, cuFFT, and cuBLAS factories have been reset.")
    except Exception as e:
        print(e)