# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:42:30 2019

@author: Bill
"""

import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import re
import seaborn as sns


# barchart showing count of classes in subdirectories of training, validation, testing datasets.  
def barchart_TVT(training_dir, validation_dir, testing_dir, plot_property):
    plt.figure(figsize=plot_property['figsize'])
    
    title = plot_property['title']
    plot_property['title'] = title + " Training"
    subplot_no = plot_property['subplot'] 

    cnt_bar_plot(training_dir, plot_property)
    
    
    plot_property['title'] = title + " Validation"
    plot_property['subplot'] = subplot_no+1
    cnt_bar_plot(validation_dir, plot_property)
    
    
    plot_property['title'] = title + " Testing"
    plot_property['subplot'] = subplot_no + 2
    cnt_bar_plot(testing_dir, plot_property)
    
    plt.show()
    

def get_subplot_params(nrows, ncols, dpi):
    subplot_params = {}
    
    subplot_params["nrows"] = nrows
    subplot_params["ncols"] = ncols
    subplot_params["figsize_col"] = subplot_params["ncols"]*2.5
    subplot_params["figsize_row"] = subplot_params["nrows"]*2.5
    subplot_params["dpi"] = dpi
    subplot_params["facecolor"] = 'w'
    subplot_params["edgecolor"] = 'k'
    subplot_params["subplot_kw"] = {'xticks': [], 'yticks': []}
    subplot_params["axes.titlesize"] = 'small'
    subplot_params["hspace"] = 0.5
    subplot_params["wspace"] = 0.3
    
    return subplot_params



def get_plot_params(figsize=(15, 5), title="", xlabel ="", ylabel="", legends=[], title_fontsize = 10, label_fontsize = 8, image_file_name="", save = False, dpi=100, update_image=True):
    plot_params = {}
    
    plot_params["figsize"] = figsize    
    plot_params["title"] = title    
    plot_params["xlabel"] = xlabel
    plot_params["ylabel"] = ylabel    
    plot_params["legends"] = legends     
    plot_params["title_fontsize"] = title_fontsize
    plot_params["axes.titlesize"] = "small"
    plot_params["label_fontsize"] = label_fontsize
    plot_params["image_file_name"] = image_file_name
    plot_params["save"] = save
    plot_params["update_image"] = update_image    
    plot_params["subplot"] = None
    
    return plot_params


# count number of files in each subdirectory of a directory
def subdir_file_cnt(root_directory):
    subdirs = os.listdir(root_directory)

    subdir_names = []
    subdir_file_cnts = []

    for subdirectory in subdirs:
        current_directory = os.path.join(root_directory, subdirectory)
        file_cnt = len(os.listdir(current_directory))
        subdir_names.append(subdirectory)
        subdir_file_cnts.append(file_cnt)
    
    return subdir_names, subdir_file_cnts


# set barplot properties.
def bar_plot(x, y, plot_property):
    if plot_property['subplot']:
        plt.subplot(plot_property['subplot'])
    sns.barplot(x=x, y=y)
    plt.title(plot_property['title'], fontsize=plot_property['title_fontsize'])
    plt.xlabel(plot_property['xlabel'], fontsize=plot_property['label_fontsize'])
    plt.ylabel(plot_property['ylabel'], fontsize=plot_property['label_fontsize'])
    plt.xticks(range(len(x)), x)
    
    
# get barplot count of labels.
def cnt_bar_plot(root_directory, plot_property):
    dir_name, dir_file_cnt = subdir_file_cnt(root_directory)
    x = [clean_stringdata(i) for i in dir_name]        
    y = dir_file_cnt
    bar_plot(x, y, plot_property)
    
    
# cleans string data.
def clean_stringdata(name):
    return re.sub(r'[^a-zA-Z,:]', ' ', name).title()

