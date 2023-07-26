import os.path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
import matplotlib.ticker as mtick
#sns.set()
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
plt.style.use('classic')
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['legend.title_fontsize'] = BIGGER_SIZE


def CombineResults():
    # root_dir=f'./results_Magzine/closed-form/e50_lamda0.0001_lr0.1_sclTrue/'
    # root_dir = f'./results_Magzine/alternative/e50_lamda1e-06_lr1.0_sclFalse/'
    # root_dir = f"./results_AuthorPapers/closed-form/e50_lamda1e-06_lr1.0_sclTrue/"
    # root_dir = f'./results_AuthorPapers/closed-form/e40_lamda0.001_lr0.01_sclFalse/'
    # root_dir = f"./results_AuthorPapers/alternative/e50_lamda1e-06_lr1.0_sclFalse/"
    # root_dir = f"./results_Magzine/random/e50_lamda1e-06_lr1.0_sclFalse/"
    # root_dir = f'./results_AuthorPapers/DeepWalk/e60_lamda1e-06_lr1.0_sclFalse/'
    root_dir = f'./results_Magzine/DeepWalk/e60_lamda1e-06_lr1.0_sclFalse/'
    files=[]
    for repeat in [2021,2022,2023,2024,2025,2026,2027,2028,2029,2030]:
        files.append(root_dir+f'bnode_degree_{repeat}/attack_graph_results.pkl')


    df_combine = pd.DataFrame()
    df_melted_combine = pd.DataFrame()
    df_evasion_top1 = pd.DataFrame()
    df_evasion_top5 = pd.DataFrame()
    df_evasion_top10 = pd.DataFrame()
    decimal = 1

    for f in files:
        try:
            f_n = open(f, 'rb')
            data = pickle.load(f_n)
        except:
            print(f'file not found:{f}')
            continue

        df_n = data
        print(df_n)

        df_evasion_top1 = df_n["detected_top1%"].map(lambda x: 1 - x)
        df_evasion_top5 = df_n["detected_top5%"].map(lambda x: 1 - x)
        df_evasion_top10 = df_n["detected_top10%"].map(lambda x: 1 - x)
        df_n.insert(data.shape[1], "detect top1%", df_evasion_top1)
        df_n.insert(data.shape[1], "detect top5%", df_evasion_top5)
        df_n.insert(data.shape[1], "detect top10%", df_evasion_top10)


        df_melted = pd.melt(df_n,
                            id_vars=['budget'],
                            value_vars=['budget', 'detect top1%', 'detect top5%', 'detect top10%'],
                            var_name=["threshold"],
                            value_name='data')

        df_combine = pd.concat([df_combine, df_n], ignore_index=True)
        df_melted_combine = pd.concat([df_melted_combine, df_melted], ignore_index=True)

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='avarage score', data=df_combine, legend='full', linewidth=2.0)
    plt.ylabel('average anomaly score')
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4e'))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    if decimal >= 3:
        plt.xticks(rotation=45)
    plt.savefig(root_dir+'BipartieGraphAttackResult_anomaly_score_combine.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='ranking', data=df_combine, legend='full', linewidth=2.0)
    plt.ylabel('anomaly ranking')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    if decimal >= 3:
        plt.xticks(rotation=45)
    plt.savefig(root_dir+'BipartieGraphAttackResult_ranking_combine.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='data', hue='threshold', data=df_melted_combine, legend='full',
                 linewidth=2.0)
    plt.ylabel('evasion successful rate')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    if decimal >= 3:
        plt.xticks(rotation=45)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.legend(loc='lower right')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    plt.savefig(root_dir+'BipartieGraphAttackResult_evasionRate_combine.pdf', dpi=300)
    plt.show()


    df_temp=df_melted_combine.query("threshold == 'detect top5%'")
    print(df_temp.groupby(['budget'])['data'].agg('mean'))
    df_temp=df_melted_combine.query("threshold == 'detect top10%'")
    print(df_temp.groupby(['budget'])['data'].agg('mean'))

    df_temp = df_combine.groupby(['budget'])['avarage score'].agg('mean')
    ori=df_temp[0]
    df_temp=((df_temp-df_temp[0])/ori) * 100
    print(df_temp)


if __name__ == "__main__":
    CombineResults()
