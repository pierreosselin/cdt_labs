"""Module for Util Functions"""
import numpy as np

def transform_time_day(list_el):
    def _aux(el):
        sp = el.split("T")
        date = sp[0]
        time = sp[1].split(":")
        day = int(date.split("-")[-1])
        time = (int(time[0]) + int(time[1])/60 + int(time[2]))/24
        time = day + time
        return time

    list_time = [_aux(el) for el in list_el]
    return (np.array(list_time) - list_time[0])

def transform_time_hour(list_el):
    def _aux(el):
        sp = el.split("T")
        date = sp[0]
        time = sp[1].split(":")
        day = int(date.split("-")[-1])
        time = int(time[0]) + int(time[1])/60 + int(time[2])
        time = day*24 + time
        return time

    list_time = [_aux(el) for el in list_el]
    return (np.array(list_time) - list_time[0])

def extend_list(index_trainable, initial_parameters):
    def extend(l):
        initial_parameters[index_trainable] = l
        return initial_parameters
    return extend
