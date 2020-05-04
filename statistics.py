import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_source_cols(df, rafd=False):
    """
    Given a dataframe df, create all necessary source columns.
    
    The input dataframe should have a column named source_class from which each row
    has a name in the form of
        'class_sess_pose_illum_expr_pitch_yaw_roll.extension'

    Alternatively, if the input is specifically a RaFD image, process it in the form of
        'pose_class_ethnicity_gender_mood_gaze'

    The output is a copy of the original dataframe with added columns.

    NOTE: the if/else is required because targets can have a name of '-1_'.
    """
    df = df.copy()

    if rafd:
        for name in ['source_name', 'target_name']:
            n = name.split('_')[0] + '_'

            df[n + 'pose'] = df[name].apply(lambda x: int(x.split('_')[0]) if (x != '-1_') else None)
            df[n + 'class'] = df[name].apply(lambda x: int(x.split('_')[1]) if (x != '-1_') else None)

            # I'm not quite sure if we will use gender or ethnicity, but it might be good to have
            df[n + 'ethnicity'] = df[name].apply(lambda x: x.split('_')[2] if (x != '-1_') else None)
            df[n + 'gender'] = df[name].apply(lambda x: x.split('_')[3] if (x != '-1_') else None)
            df[n + 'expression'] = df[name].apply(lambda x: x.split('_')[4] if (x != '-1_') else None)
            df[n + 'gaze'] = df[name].apply(lambda x: x.split('_')[5].split('.')[0] if (x != '-1_') else None)
        return df
    else:
        for name in ['source_name', 'target_name']:
            n = name.split('_')[0] + '_'

            df[n + 'class'] = df[name].apply(lambda x: int(x.split('_')[0]) if (x != '-1_') else None)
            df[n + 'session'] = df[name].apply(lambda x: float(x.split('_')[1]) if (x != '-1_') else None)

            df[n + 'pose'] = df[name].apply(lambda x: float(x.split('_')[2]) if (x != '-1_') else None)
            df[n + 'illumination'] = df[name].apply(lambda x: float(x.split('_')[3]) if (x != '-1_') else None)
            df[n + 'expression'] = df[name].apply(lambda x: float(x.split('_')[4]) if (x != '-1_') else None)


            df[n + 'pitch'] = df[name].apply(lambda x: float(x.split('_')[5][1:]) if (x != '-1_') else None)
            df[n + 'yaw'] = df[name].apply(lambda x: float(x.split('_')[6][1:]) if (x != '-1_') else None)
            df[n + 'roll'] = df[name].apply(lambda x: float(x.split('_')[7].split('.')[0][1:]) if (x != '-1_') else None)

            # Illumination changed, modify this and uncomment the previous roll
            df[n + 'roll'] = df[name].apply(lambda x: float(x.split('_')[7][1:]) if (x != '-1_') else None)
            df[n + 'illum_augmented'] = df[name].apply(lambda x: float(x.split('_')[8][2:]) if (x != '-1_') else None)
            df[n + 'intensity_augmented'] = df[name].apply(lambda x: float(x.split('_')[8].split('.')[0][2:]) if (x != '-1_') else None)
    return df


def statistics(df, dfname, filewriter, beta=1):
    # Calculate true positive, false positive, true negative and false negative respectively.
    tp = len(df.loc[(df['correct']) & (df['target_name'] != '-1_')])  # How much did it correctly classify 
    tn = len(df.loc[(df['correct'])  & (df['target_name'] == '-1_')])  # Correctly classified as unknown
    fp = len(df.loc[(~df['correct']) & (df['target_name'] != '-1_')])
    fn = len(df.loc[(~df['correct']) & (df['target_name'] == '-1_')])

    accuracy  = df['correct'].mean()
    recall    = tp / (tp + fn)
    precision = tp / (tp + fp)

    # For false positives etc we need only concern ourselves with wrong classifications
    def F(beta):
        return (1 + beta**2) * (precision * recall / (beta**2 * precision + recall + 1e-5))

    assert tp + tn + fp + fn == len(df), 'helemaal space man 🛸'
    filewriter.write(f"| {dfname} | {accuracy:.5f} | {precision:.5f} | {recall:.5f} | {F(beta):.5f} | {tp} | {tn} | {fp} | {fn} |  \n")


def create_histograms(df, fname, folder_loc, dpi=100, rafd=False):
    false_pos = df.loc[(~df['correct']) & (df['target_name'] != '-1_')]
    false_neg = df.loc[(~df['correct']) & (df['target_name'] == '-1_')]
    plt.figure(figsize=(21, 9))


    sns.distplot(false_pos['source_class'].astype(int),
             norm_hist=False,
             kde=False,
             bins=false_pos['source_class'].nunique(),
             label=f"False Positives, num = {false_pos['source_class'].count()}"
            )
    sns.distplot(false_neg['source_class'].astype(int),
             norm_hist=False,
             kde=False,
             bins=false_neg['source_class'].nunique(),
             label=f"False Negatives, num = {false_neg['source_class'].count()}"
            )

    plt.xlabel("Source Class", fontsize=20) ; plt.ylabel('Number of Errors', fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(f"{folder_loc}/{fname}_falsepos.png", dpi=dpi)


def pose_illum_express(df, fname, folder_loc):
    false_neg = df.loc[(~df['correct']) & (df['target_name'] == '-1_')]
    plt.figure(figsize = (20, 6))

    plt.subplot(131)  # EXPRESSION
    uniqs_expres = np.array([0, 1, 2])  #sorted(false_neg['source_expression'].astype(int).unique())
    expres_dict = {uniq.astype(int): sum(false_neg['source_expression'].astype(int) == uniq) for uniq in uniqs_expres}
    plt.bar(x = range(len(uniqs_expres)), height = expres_dict.values(), alpha=0.5)
    plt.xticks(range(len(uniqs_expres)), uniqs_expres)
    plt.xlabel("Expression type", fontsize=20) ; plt.ylabel("Number of Errors", fontsize=20);

    plt.subplot(132)  # POSE
    uniqs_pose = np.array([80, 130, 140, 51, 50, 41, 190])
    pose_dict = {uniq.astype(int): sum(false_neg['source_pose'].astype(int) == uniq) for uniq in uniqs_pose}
    plt.bar(x = range(len(uniqs_pose)), height = pose_dict.values(), alpha=0.5)
    plt.xticks(range(len(uniqs_pose)), uniqs_pose)
    plt.xlabel("Pose angle", fontsize=20) ; plt.ylabel("Number of Errors", fontsize=20);

    plt.subplot(133)  # ILLUMINATION
    uniqs_illum = np.array([2, 7, 17, 12])
    illumination_dict = {uniq.astype(int): sum(false_neg['source_illumination'].astype(int) == uniq) for uniq in uniqs_illum}
    plt.bar(x = range(len(uniqs_illum)), height = illumination_dict.values(), alpha=0.5)
    plt.xticks(range(len(uniqs_illum)), uniqs_illum)
    plt.xlabel("Illumination type", fontsize=20) ; plt.ylabel("Number of Errors", fontsize=20);

    plt.savefig(f"{folder_loc}/{fname}_hist_exp_pose_ill.png", dpi=100)


def hist_joint(dfs, fnames, folder_loc, rafd=False):
    new_dfs = []
    for df, fname in zip(dfs, fnames):
        df['name'] = fname
        df = df.loc[(~df['correct']) & (df['target_name'] == '-1_')]
        new_dfs.append(df)
    df = pd.concat(new_dfs)

    if rafd:

        plt.figure(figsize=(21, 6))
        plt.subplot(1, 2, 1)
        sns.countplot(x='source_expression', hue='name', data=df)
        plt.subplot(1, 2, 2)
        sns.countplot(x='source_pose', hue='name', data=df) # Cant add order yet because
        # sometimes the CSVs have less poses because we only focus on frontal
        plt.savefig(f'{folder_loc}/histogram_joint_plot.png', dpi=200)
        return

    plt.figure(figsize=(21, 6))
    plt.subplot(1, 3, 1)
    sns.countplot(x='source_expression', hue='name', data=df)
    plt.subplot(1, 3, 2)
    sns.countplot(x='source_pose', hue='name', data=df, order=[80, 130, 140, 51, 50, 41, 190])
    plt.subplot(1, 3, 3)
    sns.countplot(x='source_illumination', hue='name', data=df, order=[2, 7, 17, 12])
    plt.savefig(f'{folder_loc}/histogram_joint_plot.png', dpi=200)

def write_to_markdown(dfs, fnames, file_loc, folder_loc, rafd):
    """
    dfs: list of dataframes [df_1, ..., df_n]
    fnames: list of csv names [cvname_1, ..., cvname_n]
    file_loc: whatever.md (specifically markdown)
    """
    filewriter = open(folder_loc + '/' + file_loc, 'w')
    filewriter.write("# Output statistics\n")
    filewriter.write("## Table of Statistics\n")
    filewriter.write('| fname | Accuracy | Precision | Recall | F-score | True Positive | True Negative | False Positive | False Negative |\n')
    filewriter.write('|--------|----------|-----------|--------|---------|---------------|---------------|----------------|----------------|\n')

    # Write statistics for each CSV file
    for df, fname, in zip(dfs, fnames):
        statistics(df, fname, filewriter)
    filewriter.write('\n')

    # Create fancy images and write to files
    filewriter.write("## Histogram of False Positives\n")
    if not rafd:
        for df, fname in zip(dfs, fnames):
            filewriter.write(f"### {fname}\n")
            create_histograms(df, fname, folder_loc, rafd=rafd)
            filewriter.write(f"<img src='{fname}_falsepos.png'>\n")

    filewriter.write("## Pose, Expressions & Illumination\n")
    filewriter.write("### Joint plot\n")
    hist_joint(dfs, fnames, folder_loc, rafd=rafd)
    filewriter.write("<img src='histogram_joint_plot.png'>\n")
    if rafd:
        dfs_cat = pd.concat(dfs)
        plt.figure(figsize=(16, 16))
        plt.subplot(2, 3, 1)
        sns.countplot(x='source_expression', hue='name', data=dfs_cat)
        plt.subplot(2, 3, 2)
        sns.countplot(x='source_gender', hue='name', data=dfs_cat)
        plt.subplot(2, 3, 3)
        sns.countplot(x='source_pose', hue='name', data=dfs_cat)
        plt.subplot(2, 3, 4)
        sns.countplot(x='source_gaze', hue='name', data=dfs_cat)
        plt.subplot(2, 3, 5)
        sns.countplot(x='source_ethnicity', hue='name', data=dfs_cat)
        plt.subplot(2, 3, 6)
        sns.countplot(x='source_class', hue='name', data=dfs_cat)
        plt.savefig(f"{folder_loc}/multiplot.png", dpi=100)
        filewriter.write("<img src='multiplot.png'>\n")
    else:
        for df, fname in zip(dfs, fnames):
            filewriter.write(f"### {fname}\n")
            pose_illum_express(df, fname, folder_loc)
            filewriter.write(f"<img src='{fname}_hist_exp_pose_ill.png'>\n")

def write_single_csv(input_csv, output_loc, rafd):
    csv_name = "".join(input_csv.split('/')[-2:])
    df = pd.read_csv(input_csv)
    df = create_source_cols(df, rafd)
    write_to_markdown([df], [csv_name], output_loc, rafd)

def write_multiple_csv(multiple_csv_loc, output_loc, rafd):
    fs = os.listdir(multiple_csv_loc)
    fnames = []
    dfs = []
    for f in fs:
        df = pd.read_csv(multiple_csv_loc + '/' + f)
        fnames.append("_".join('.'.join(f.split('.')[0:2]).split('_')[1:]))
        dfs.append(create_source_cols(df, rafd))
    write_to_markdown(dfs, fnames, output_loc, multiple_csv_loc, rafd)


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default=None, help='location of a single CSV file to be processed')
parser.add_argument('--multiple_csv_loc', type=str, default=None)
parser.add_argument('--output_loc', type=str, default='stats.md', help='location of statistics markdown file to write to')
parser.add_argument('--rafd', type=bool, default=False, help='Enable statistics for RAFD only.')

if __name__ == '__main__':
    # Example run:
    # python statistics --multiple_csv_loc csvs/ --output_loc stats.md


    sns.set_style('whitegrid')
    args = parser.parse_args()

    if args.input_csv:
        write_single_csv(args.input_csv, args.output_loc, rafd=args.rafd)
    if args.multiple_csv_loc:
        write_multiple_csv(args.multiple_csv_loc, args.output_loc, rafd=args.rafd)

