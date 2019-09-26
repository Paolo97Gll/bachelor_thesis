#!/usr/bin/env python3
# coding: utf-8


# PLOT IN PDF FORMAT


import pandas as pd
import matplotlib.pyplot as plt

with pd.HDFStore('ris/OUT-classified.h5') as data:
    for n in ['101', '227', '22', '245', '29', '408', '42', '48', '4', '66', '76', '8']:
        CHOICE = '/GLITCH/' + str(n)
        plt.plot(data[CHOICE], marker='.', linestyle='dashed')
        plt.title('Glitch ' + str(n))
        plt.xlabel('Time [s]')
        plt.ylabel('Signal [T_cmb V / W]')
        plt.savefig('ris/plots/glitch-' + str(n) + '.pdf')
        plt.close()
    for n in ['162', '182', '1', '28', '33', '34', '54', '87', '93']:
        CHOICE = '/MULTI_GLITCH/' + str(n)
        plt.plot(data[CHOICE], marker='.', linestyle='dashed')
        plt.title('Multi glitch ' + str(n))
        plt.xlabel('Time [s]')
        plt.ylabel('Signal [T_cmb V / W]')
        plt.savefig('ris/plots/multi_glitch-' + str(n) + '.pdf')
        plt.close()
    for n in ['100', '10', '14', '423', '427', '428']:
        CHOICE = '/NO_GLITCH/' + str(n)
        plt.plot(data[CHOICE], marker='.', linestyle='dashed')
        plt.title('No glitch ' + str(n))
        plt.xlabel('Time [s]')
        plt.ylabel('Signal [T_cmb V / W]')
        plt.savefig('ris/plots/no_glitch-' + str(n) + '.pdf')
        plt.close()