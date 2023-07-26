import os.path
import pickle
import pandas as pd
import numpy as np
import argparse
import pprint
import collections
import matplotlib.pyplot as plt
import seaborn as sns;
import matplotlib.ticker as mtick
import seaborn as sns;
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
plt.style.use('classic')
MEDIUM_SIZE = 19
BIGGER_SIZE = 21
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['legend.title_fontsize'] = BIGGER_SIZE
decimal=1
def collect_data(dir,file_name):
    files = []
    for repeat in [2021,2022,2023,2024,2025,2026,2027,2028,2029,2030]:  #
        files.append(dir + file_name.format(repeat=repeat))
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
            print(f"missing: {f}")
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
    return df_combine,df_anomaly_combine,df_melted_combine


def CombineResults(args):
    # palette = ["#CA3705", "#53A344","#5779C1"]
    dataset="Mnist"#KDD99,Mnist
    root_dir_graph = f'/home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Proximity/01_attack_graph/results_{dataset}/'
    root_dir_feature=f'/home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Proximity/02_attack_feature/results_{dataset}/'

    if dataset == "Mnist":
        args.apply_constraint=False
        output_dir = root_dir_feature + f"Combined_c{args.apply_constraint}/"
        #args.budget_mode = "attack_number"
        # graph attack results
        graph_dir = root_dir_graph + f'alternative/e100_lamda0.0001_lr0.01_sFalse/'
        # feature attack: random
        dir1 = root_dir_feature + f"random/target_anomaly/e500_ld0.0_lr1.0_c{args.apply_constraint}_b{args.budget_mode}_sFalse/"
        # feature attack: graph guided with L_a
        dir2 = root_dir_feature + f"graph_guided/target_anomaly/e500_ld0.0_lr1.0_c{args.apply_constraint}_b{args.budget_mode}_sFalse/"
        # feature attack: graph guided with L_a cf
        dir3 = root_dir_feature + f"graph_guided_cf/target_anomaly/e500_ld0.0_lr1.0_c{args.apply_constraint}_b{args.budget_mode}_sFalse/"
        # feature attack: graph guided with L_g
        dir4 = root_dir_feature + f"graph_guided/attacked_graph/e1000_ld0.0_lr1.0_c{args.apply_constraint}_b{args.budget_mode}_sFalse/"
        df_1, df_anomaly_1, df_melted_1 = collect_data(dir1,'{repeat}/attack_feature_results.pkl')
        df_2, df_anomaly_2, df_melted_2 = collect_data(dir2,'{repeat}/attack_feature_results.pkl')
        df_3, df_anomaly_3, df_melted_3 = collect_data(dir3,'{repeat}/attack_feature_results.pkl')
        df_4, df_anomaly_4, df_melted_4 = collect_data(dir4, '{repeat}/attack_feature_results.pkl')
        df_g, df_anomaly_g, df_melted_g = collect_data(graph_dir, 'target20_thre0.5_{repeat}/attack_graph_results.pkl')
        df_melted_g=df_melted_g[(df_melted_g["budget"]!=0.05) & (df_melted_g["budget"]!=0.1)]

    elif dataset == "KDD99":#
        output_dir = root_dir_feature + f"Combined_c{args.apply_constraint}/"
        graph_dir=root_dir_graph+f'alternative/e35_lamda0.0001_lr1.0_sFalse/'
        dir1 = root_dir_feature+f"e300_ld0.0_lr1.0_mrandom_c{args.apply_constraint}/"
        dir2 = root_dir_feature+f"e300_ld0.0_lr1.0_mgraph_guided_c{args.apply_constraint}/"
        # feature attack: graph guided with L_a cf
        dir3 = root_dir_feature + f"graph_guided_cf/target_anomaly/e500_ld0.0_lr1.0_c{args.apply_constraint}_b{args.budget_mode}_sFalse/"
        dir4 = root_dir_feature+f"e500_ld0.0_lr10.0_mgraph_guided_c{args.apply_constraint}/"
        df_1, df_anomaly_1, df_melted_1 = collect_data(dir1,'bnode_degree_altarget_anomaly_sFalse_{repeat}/attack_feature_results.pkl')
        df_2, df_anomaly_2, df_melted_2 = collect_data(dir2,'bnode_degree_altarget_anomaly_sFalse_{repeat}/attack_feature_results.pkl')
        df_3, df_anomaly_3, df_melted_3 = collect_data(dir3, '{repeat}/attack_feature_results.pkl')
        df_4, df_anomaly_4, df_melted_4 = collect_data(dir4,'bnode_degree_alattacked_graph_sFalse_{repeat}/attack_feature_results.pkl')
        df_g, df_anomaly_g, df_melted_g = collect_data(graph_dir, 'target20_thre0.8_{repeat}/attack_graph_results.pkl')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_1["attack types"] = "VanillaOpt"
    df_anomaly_1["attack types"] = "VanillaOpt"
    df_melted_1["attack types"] = "VanillaOpt"
    df_2["attack types"] = "G-guided-alterI"
    df_anomaly_2["attack types"] = "G-guided-alterI"
    df_melted_2["attack types"] = "G-guided-alterI"
    df_3["attack types"] = "G-guided-cf"
    df_anomaly_3["attack types"] = "G-guided-cf"
    df_melted_3["attack types"] = "G-guided-cf"
    df_4["attack types"] = "G-guided-plus"
    df_anomaly_4["attack types"] = "G-guided-plus"
    df_melted_4["attack types"] = "G-guided-plus"
    if args.budget_mode!="attack_number":
        df_g["attack types"] = "graph attack"
        df_anomaly_g["attack types"] = "graph attack"
        df_melted_g["attack types"] = "graph attack"
        df_anomaly_combine = pd.concat([df_anomaly_1, df_anomaly_2, df_anomaly_g], ignore_index=True)
        df_combine = pd.concat([df_1, df_2, df_g], ignore_index=True)
        df_melted_combine = pd.concat([df_melted_1, df_melted_2, df_melted_g], ignore_index=True)
        df_anomaly_combine = pd.concat([df_anomaly_1, df_anomaly_2, df_anomaly_3, df_anomaly_4,df_anomaly_g], ignore_index=True)
        df_combine = pd.concat([df_1, df_2, df_3, df_4, df_g], ignore_index=True)
        df_melted_combine = pd.concat([df_melted_1, df_melted_2, df_melted_3, df_melted_4,df_melted_g], ignore_index=True)
        palette = ["#ff7878","#a2c4c9","#f9cb9c","#3d85c6","#a7adba"]# blue:"#3d85c6" green: b2d8d8
        df_anomaly_combine = df_anomaly_combine[
            (df_anomaly_combine["budget"] != 0.05) & (df_anomaly_combine["budget"] != 0.1)]
        df_combine = df_combine[
            (df_combine["budget"] != 0.05) & (df_combine["budget"] != 0.1)]
        df_melted_combine = df_melted_combine[
            (df_melted_combine["budget"] != 0.05) & (df_melted_combine["budget"] != 0.1)]
    else:
        df_anomaly_combine = pd.concat([df_anomaly_1, df_anomaly_2], ignore_index=True)
        df_combine = pd.concat([df_1, df_2], ignore_index=True)
        df_melted_combine = pd.concat([df_melted_1, df_melted_2], ignore_index=True)
        palette = ["#ff7878", "#a2c4c9"]

    df_temp=df_combine[(df_combine["similarity"] == "cosine")]
    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='average_anomaly_score', hue='attack types', data=df_temp, legend='full', linewidth=2.0,palette=palette)
    plt.ylabel('anomaly score')
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4e'))
    if args.budget_mode!="attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.legend(loc='lower left',fancybox=True, framealpha=0.5)
    plt.savefig(output_dir + f'Contrast_anomaly_score_constraint{args.apply_constraint}.pdf',dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='ranking', hue='attack types', data=df_temp,
                 legend='full', linewidth=2.0,palette=palette)
    plt.ylabel('anomaly ranking')
    plt.legend(loc='upper left',fancybox=True, framealpha=0.5)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(output_dir + f'Contrast_ranking_constraint{args.apply_constraint}.pdf', dpi=300)
    plt.show()

    df_cor = df_melted_combine[(df_melted_combine["similarity"] == "correlation")&(df_melted_combine["threshold"] == "detect top5%")]
    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='data', hue="attack types", markers=True, data=df_cor,
                 legend='full',linewidth=2.0,palette=palette)
    plt.ylabel('evasion rate')
    plt.legend(loc='upper left',fancybox=True, framealpha=0.5)
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(output_dir + f'Contrast_evasion_constraint{args.apply_constraint}_correlation.pdf', dpi=300)
    plt.show()

    if args.budget_mode != "attack_number":
        df_cor['budget'] = [str(round(x * 100, decimal)) + "%" for x in df_cor['budget']]
    else:
        df_cor['budget'] = [str(x) for x in df_cor['budget']]

    plt.figure(constrained_layout=True)
    p = sns.barplot(x='budget', y='data', hue="attack types", data=df_cor,palette=palette)
    for bar in p.containers[4]:
        bar.set_alpha(0.7)
        bar.set_linestyle('-.')
    plt.legend(loc='upper left', fancybox=True, framealpha=0.5)
    plt.ylabel('evasion rate (top 5%)')
    plt.savefig(output_dir + f'Contrast_evaTop5_constraint{args.apply_constraint}_correlation.pdf', dpi=300)
    plt.show()

    df_cos = df_melted_combine[(df_melted_combine["similarity"] == "cosine")&(df_melted_combine["threshold"] == "detect top5%")]
    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='data', hue="attack types", markers=True, data=df_cos,
                 legend='full',
                 linewidth=2.0,palette=palette)
    plt.ylabel('evasion rate')
    plt.legend(loc='upper left',fancybox=True, framealpha=0.5)
    plt.savefig(output_dir + f'Contrast_evasion_constraint{args.apply_constraint}_cosine.pdf', dpi=300)
    plt.show()

    if args.budget_mode != "attack_number":
        df_cos['budget'] = [str(round(x * 100, decimal)) + "%" for x in df_cos['budget']]
    else:
        df_cos['budget'] = [str(x) for x in df_cos['budget']]
    plt.figure(constrained_layout=True)
    p = sns.barplot(x='budget', y='data', hue="attack types", data=df_cos,palette=palette)
    for bar in p.containers[4]:
        bar.set_alpha(0.7)
        bar.set_linestyle('-.')
    plt.legend(loc='upper left', fancybox=True, framealpha=0.5)
    plt.ylabel('evasion rate (top 5%)')
    plt.savefig(output_dir + f'Contrast_evaTop5_constraint{args.apply_constraint}_cosine.pdf', dpi=300)
    plt.show()

    print("results (figures) saved to :",output_dir)


def load_data(dir,file):
    f = open(dir + file, 'rb')
    df, df_degrees = pickle.load(f)
    f.close()
    return df, df_degrees


def CombineAnaylsis(args):
    dataset = "Mnist"  # KDD99,Mnist
    root_dir_feature = f'/home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Proximity/02_attack_feature/results_{dataset}/'

    if dataset == "Mnist":
        args.apply_constraint=False
        output_dir = root_dir_feature + f"Combined_c{args.apply_constraint}/"
        #args.budget_mode = "attack_number"
        # feature attack: random
        dir1 = root_dir_feature + f"random/target_anomaly/e500_ld0.0_lr1.0_c{args.apply_constraint}_b{args.budget_mode}_sFalse/"
        # feature attack: graph guided with L_a
        dir2 = root_dir_feature + f"graph_guided/target_anomaly/e500_ld0.0_lr1.0_c{args.apply_constraint}_b{args.budget_mode}_sFalse/"
        # feature attack: graph guided with L_a cf
        dir3 = root_dir_feature + f"graph_guided_cf/target_anomaly/e500_ld0.0_lr1.0_c{args.apply_constraint}_b{args.budget_mode}_sFalse/"
        df_1, df_degrees_1 = load_data(dir1,'2024/attack_feature_analysis_cosine.pkl')
        df_2, df_degrees_2 = load_data(dir2,'2024/attack_feature_analysis_cosine.pkl')
        df_3, df_degrees_3 = load_data(dir3,'2024/attack_feature_analysis_cosine.pkl')

    elif dataset == "KDD99":#
        output_dir = root_dir_feature + f"Combined_c{args.apply_constraint}/"
        dir1 = root_dir_feature+f"e300_ld0.0_lr1.0_mrandom_c{args.apply_constraint}/"
        dir2 = root_dir_feature+f"e300_ld0.0_lr1.0_mgraph_guided_c{args.apply_constraint}/"
        # feature attack: graph guided with L_a cf
        dir3 = root_dir_feature + f"graph_guided_cf/target_anomaly/e500_ld0.0_lr1.0_c{args.apply_constraint}_b{args.budget_mode}_sFalse/"
        df_1, df_degrees_1 = load_data(dir1,'bnode_degree_altarget_anomaly_sFalse_2024/attack_feature_analysis_cosine.pkl')
        df_2, df_degrees_2 = load_data(dir2,'bnode_degree_altarget_anomaly_sFalse_2024/attack_feature_analysis_cosine.pkl')
        df_3, df_degrees_3 = load_data(dir3, '2024/attack_feature_analysis_cosine.pkl')


    df_1["attack types"] = "VanillaOpt"
    df_2["attack types"] = "graph-guided-alterI"
    df_3["attack types"] = "graph-guided-plus"
    df_degrees_1["attack types"] = "VanillaOpt"
    df_degrees_2["attack types"] = "graph-guided"
    df_combine = pd.concat([df_1, df_2, df_3], ignore_index=True)
    df_degrees_combine = pd.concat([df_degrees_1, df_degrees_2], ignore_index=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(constrained_layout=True)
    sns.barplot(x='budget', y='edge_modified',hue="attack types", data=df_combine, palette="Blues")
    plt.ylabel('edge modified (feature attack)')
    plt.xlabel('edge modified (graph attack)')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', fancybox=True, framealpha=0.5)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    plt.savefig(output_dir + f'Contrast_AnalysisEdge.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.boxplot(x='budget', y='control_degrees',hue="attack types", data=df_degrees_combine, palette="Greens")
    plt.ylabel('control node degrees')
    plt.legend(loc='upper left', fancybox=True, framealpha=0.5)
    plt.savefig(output_dir + f'Contrast_AnalysisDegree.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attack RW model')
    parser.add_argument('-budget_mode', type=str, default='node_degree',
                        choices=['totoal_edges', 'node_degree', 'attack_number'],
                        help="totoal_edges: total budget = budget * total edges; node_degree : total budget= budget * target nodes * average degree of target node; attack_number: number of attack ndoes")
    parser.add_argument('-apply_constraint', action='store_true', default=True,
                        help="apply features constraints if True")
    parser.add_argument('-dataset', type=str, default='MNIST', choices=['KDD99', 'MNIST'])
    args = parser.parse_args()
    CombineResults(args)
    CombineAnaylsis(args)

