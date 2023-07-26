import os.path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
import matplotlib.ticker as mtick
import seaborn as sns;
from matplotlib.patches import Rectangle
sns.set()
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
plt.style.use('classic')
MEDIUM_SIZE = 20
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['legend.title_fontsize'] = BIGGER_SIZE

dataset="MNIST"#["KDD99","MNIST"]
attack_mode="DeepWalk" #['alternative', 'closed-form','random','degree','DeepWalk']
def CombineResults():
    if dataset=="KDD99":
        if attack_mode in ["alternative","DeepWalk"]:
            root_dir = f"./results_KDD99/{attack_mode}/e35_lamda0.0001_lr1.0_sFalse/"
        else:
            root_dir = f"./results_KDD99/{attack_mode}/e35_lamda0.0001_lr0.1_sFalse/"
        foldername="target20_thre0.8"
    elif dataset=="MNIST":
        root_dir = f"./results_Mnist/{attack_mode}/e100_lamda0.0001_lr0.01_sFalse/"
        foldername = "target20_thre0.5"
    files=[]
    for repeat in [2021,2022,2023,2024,2025,2026,2027,2028, 2029, 2030]:#
        files.append(root_dir+f'{foldername}_{repeat}/attack_graph_results.pkl')

    df_combine = pd.DataFrame()
    df_anomaly_combine = pd.DataFrame()
    df_melted_combine = pd.DataFrame()
    decimal = 1

    for f in files:
        try:
            f_n = open(f, 'rb')
            data = pickle.load(f_n)
            f_n.close()
        except:
            print(f'missing:{f}')
            continue
        df_n = data[0]
        df_anomaly_n = data[1]
        df_melted_n = data[2]
        average_score = []

        for i in range(len(df_n)):
            average_score.append(df_n["anomaly score"][i].mean())
        df_n.insert(df_n.shape[1], 'average_anomaly_score', average_score)

        df_combine = pd.concat([df_combine, df_n], ignore_index=True)
        df_anomaly_combine = pd.concat([df_anomaly_combine, df_anomaly_n], ignore_index=True)
        df_melted_combine = pd.concat([df_melted_combine, df_melted_n], ignore_index=True)

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='average_anomaly_score', hue='similarity', data=df_combine, legend='full', linewidth=2.0)
    plt.ylabel('average anomaly score')
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4e'))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    if decimal >= 3:
        plt.xticks(rotation=45)
    plt.savefig(root_dir+'GraphAttackResult_anomaly_score_combine.pdf', dpi=300)
    plt.show()


    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='ranking', hue='similarity', data=df_combine, legend='full', linewidth=2.0)
    plt.ylabel('anomaly ranking')
    plt.legend(loc='lower right')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    if decimal >= 3:
        plt.xticks(rotation=45)
    plt.savefig(root_dir+'GraphAttackResult_ranking_combine.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='data', hue='threshold', style='similarity', data=df_melted_combine, legend='full',
                 linewidth=2.0)
    plt.ylabel('evasion successful rate')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    if decimal >= 3:
        plt.xticks(rotation=45)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.legend(loc='lower right')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    plt.savefig(root_dir+'GraphAttackResult_EvasionRate_combine.pdf', dpi=300)
    plt.show()

    df_temp=df_melted_combine.query("threshold == 'detect top10%'").query("similarity=='cosine'")
    print(df_temp.groupby(['budget'])['data'].agg('mean'))
    df_temp = df_melted_combine.query("threshold == 'detect top10%'").query("similarity=='correlation'")
    print(df_temp.groupby(['budget'])['data'].agg('mean'))

    df_temp = df_anomaly_combine.query("similarity=='cosine'")
    df_temp = df_temp.groupby(['budget'])['anomaly score'].agg('mean')
    print(df_temp)
    ori=df_temp[0]
    df_temp=((df_temp-df_temp[0])/ori) * 100
    print(df_temp)

    df_temp = df_anomaly_combine.query("similarity=='correlation'")
    df_temp = df_temp.groupby(['budget'])['anomaly score'].agg('mean')
    print(df_temp)
    ori = df_temp[0]
    df_temp = ((df_temp - df_temp[0]) / ori)*100
    print(df_temp)

def CombineLoss():
    df_combined = pd.DataFrame()
    for attack_mode in ['alternative','closed-form']:
        if dataset == "KDD99":
            if attack_mode == "alternative":
                root_dir = f"/home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Proximity/01_attack_graph/results_KDD99/{attack_mode}/e300_lamda0.0001_lr1.0_sFalse/"
            else:
                root_dir = f"/home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Proximity/01_attack_graph/results_KDD99/{attack_mode}/e300_lamda0.0001_lr0.1_sFalse/"
            foldername = "target20_thre0.8"
        elif dataset == "MNIST":
            root_dir = f"/home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Proximity/01_attack_graph/results_Mnist/{attack_mode}/e300_lamda0.0001_lr0.01_sFalse/"
            foldername = "target20_thre0.5"
        df_loss = pd.DataFrame()
        for repeat in [2021]:  #, 2022, 2023, 2024, 2025, 2026, 2027, 2028
            f=root_dir + f'{foldername}_{repeat}/loss_data_correlation.pkl'
            f_n = open(f, 'rb')
            data = pickle.load(f_n)
            f_n.close()
            # data[0]['anomaly scores'].append(data[0]['anomaly scores'][-1])
            df_n = pd.DataFrame(data[0])
            df_n["seed"]=repeat
            df_loss = pd.concat([df_loss, df_n], ignore_index=True)

        name_d={'alternative':'alterI','closed-form':'cf'}
        df_loss["attack types"] = name_d[attack_mode]
        df_combined = pd.concat([df_combined, df_loss], ignore_index=True)


    plt.figure(constrained_layout=True)
    sns.lineplot(x='epochs', y='attack loss', hue='attack types',data=df_combined, legend='full',
                 linewidth=2.0)#, hue='seed'
    plt.ylabel('attack loss (continuous space)')
    plt.legend(loc='lower left', fancybox=True, framealpha=0.5)
    plt.savefig(root_dir + 'GraphAttackTrainigLossCon_combine.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='epochs', y='attack loss discrete', hue='attack types', data=df_combined, legend='full',
                 linewidth=2.0)  # , hue='seed'
    plt.ylabel('attack loss (dsicrete space)')
    plt.legend(loc='lower left', fancybox=True, framealpha=0.5)
    plt.savefig(root_dir + 'GraphAttackTrainigDis_combine.pdf', dpi=300)
    plt.show()

    df_melted = pd.melt(df_combined,
                        id_vars=['epochs', 'attack types'],
                        value_vars=['epochs', 'attack types', 'attack loss', 'attack loss approximate'],
                        var_name=["type"],
                        value_name='loss')
    df_melted = df_melted[
        (df_melted["type"] != "attack loss approximate") | (df_melted["attack types"] != "closed-form (cf)")]
    map={'attack loss':'exact','attack loss approximate':'approximate'}
    df_melted.loc[:,"type"]=[map[v] for v in df_melted["type"]]
    plt.figure(constrained_layout=True)
    sns.lineplot(x='epochs', y='loss', hue='attack types', style="type", data=df_melted, legend='full',
                 linewidth=2.0)  # , hue='seed'
    plt.ylabel('attack loss')
    plt.legend(loc='lower left', fancybox=True, framealpha=0.5)
    plt.savefig(root_dir + 'GraphAttackTrainigLossCom_combine.pdf', dpi=300)
    plt.show()

def CombineAnalysis():
    df_combined = pd.DataFrame()
    if dataset == "KDD99":
        if attack_mode == "alternative":
            root_dir = f"/home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Proximity/01_attack_graph/results_KDD99/{attack_mode}/e35_lamda0.0001_lr1.0_sFalse/"
        else:
            root_dir = f"/home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Proximity/01_attack_graph/results_KDD99/{attack_mode}/e35_lamda0.0001_lr0.1_sFalse/"
        foldername = "target20_thre0.8"
    if dataset == "MNIST":
        if attack_mode == "alternative":
            root_dir = f"/home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Proximity/01_attack_graph/results_Mnist/{attack_mode}/e100_lamda0.0001_lr0.01_sFalse/"
        foldername = "target20_thre0.5"

    for repeat in [2021,2022,2023,2024,2025,2026,2027,2028,2029,2030]:  # , 2022, 2023, 2024, 2025, 2026, 2027, 2028
        f = root_dir + f'{foldername}_{repeat}/analysis_results.pkl'
        f_n = open(f, 'rb')
        data = pickle.load(f_n)
        f_n.close()
        # data[0]['anomaly scores'].append(data[0]['anomaly scores'][-1])
        df_n = pd.DataFrame(data)
        df_n["seed"] = repeat
        df_combined = pd.concat([df_combined,df_n], ignore_index=True)

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='attack nodes', data=df_combined, legend='full', linewidth=2.0, marker='o')
    plt.ylabel('attack nodes proportion')
    decimal=2
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    #plt.xticks(rotation=45)
    plt.savefig(root_dir + 'AnalysisAttackNumber.pdf', dpi=300)
    plt.show()
    print(f'figure saved to {root_dir}')

if __name__ == "__main__":
    CombineResults()
    # CombineLoss()
    # CombineAnalysis()