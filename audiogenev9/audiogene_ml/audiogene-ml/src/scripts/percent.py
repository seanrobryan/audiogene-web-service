import numpy as np
import pandas as pd

# Functions for age groups
def age_group(i):
    r = None
    if i < 20:
        r = '20'
    if 20 <= i < 40:
        r = '20-40'
    if 40 <= i < 60:
        r = '40-60'
    if i >= 60:
        r = '60+'
    return r

def age_group_30(i):
    r = None
    if i < 30:
        r = '30'
    if 30 <= i < 60:
        r = '30-60'
    if i >= 60:
        r = '60+'
    return r

def age_group_40(i):
    r = None
    if i < 40:
        r = '40'
    if 40 <= i < 60:
        r = '40-60'
    if i >= 60:
        r = '60+'
    return r

def age_group_unique(i):
    return i

def results_age_group_summary(results_confusion_matrix, parTrueLabel, parPredictLabel, age_group_func):
    # Confusion Matrix w/ Age Groups
    results_confusion_matrix_age_group = results_confusion_matrix \
        .assign(age_group=lambda dataframe: dataframe['Age']
                .map(lambda Age: age_group_func(Age)))

    # Filter Misclassification Labels
    df_filter_result_age_group = results_confusion_matrix_age_group[
        (results_confusion_matrix_age_group['True'] == parTrueLabel) &
        (results_confusion_matrix_age_group['Predicted'] == parPredictLabel)]

    total_miss = results_confusion_matrix_age_group['age_group'][results_confusion_matrix_age_group['age_group']
        .isin(df_filter_result_age_group['age_group'].unique())] \
        .value_counts().reset_index().rename(columns={"age_group": "Total"})

    misses = df_filter_result_age_group['age_group']. \
                 value_counts() / results_confusion_matrix_age_group['age_group'][
                 results_confusion_matrix_age_group['age_group']
                     .isin(df_filter_result_age_group['age_group'].unique())].value_counts()
    missed_summary = misses.reset_index().merge(total_miss, on='index').rename(columns={"age_group": "Percentage"}) \
        .rename(columns={"index": "Age Group"}) \
        .merge(df_filter_result_age_group['age_group'].value_counts()
               .reset_index().rename(columns={"index": "Age Group", 'age_group': 'Counts'})
               , on='Age Group')

    missed_summary['Total Counts'] = missed_summary['Counts'].sum()
    missed_summary['Average Age (Misclass)'] = df_filter_result_age_group.groupby(by='age_group')['Age'].mean().tolist()
    misclass = df_filter_result_age_group['Age'].mean()
    # print(df_filter_result_age_group.groupby(by='age_group')['Age'].mean().tolist())

    # True and Predict Labels
    parPredictLabel2 = parTrueLabel

    # Filter Misclassification Labels
    df_filter_result_age_group = results_confusion_matrix_age_group[
        (results_confusion_matrix_age_group['True'] == parTrueLabel) &
        (results_confusion_matrix_age_group['Predicted'] == parPredictLabel2)]

    total_miss = results_confusion_matrix_age_group['age_group'][results_confusion_matrix_age_group['age_group']
        .isin(df_filter_result_age_group['age_group'].unique())] \
        .value_counts().reset_index().rename(columns={"age_group": "Total"})

    misses = df_filter_result_age_group['age_group']. \
                 value_counts() / results_confusion_matrix_age_group['age_group'][
                 results_confusion_matrix_age_group['age_group']
                     .isin(df_filter_result_age_group['age_group'].unique())].value_counts()
    missed_summary_classification = misses.reset_index().merge(total_miss, on='index').rename(
        columns={"age_group": "Percentage"}) \
        .rename(columns={"index": "Age Group"}) \
        .merge(df_filter_result_age_group['age_group'].value_counts()
               .reset_index().rename(columns={"index": "Age Group", 'age_group': 'Counts'})
               , on='Age Group')

    # print(missed_summary_classification)

    missed_summary_classification['Average Age (Correct)'] = df_filter_result_age_group.groupby(by='age_group')['Age'].mean().tolist()
    missed_summary_classification['Total Counts'] = missed_summary['Counts'].sum()
    missed_summary_classification_sub = missed_summary_classification[['Age Group', 'Counts', 'Average Age (Correct)']] \
        .rename(columns={'Counts': "Correct"})
    correct = df_filter_result_age_group['Age'].mean()

    missed_summary_merge = missed_summary[['Age Group', 'Total', 'Counts', 'Average Age (Misclass)']].rename(
        columns={'Counts': 'Misclass'}) \
        .merge(missed_summary_classification_sub, on='Age Group', how='outer').fillna(0).sort_values(by=['Age Group'])

    # print(missed_summary_classification_sub)

    # missed_summary_merge['Fraction (Correct vs MisClass)'] = missed_summary_merge['Correct'] / missed_summary_merge[
    #     'Misclass']
    missed_summary_merge['Total Between Two Classes'] = missed_summary_merge['Correct'] + missed_summary_merge[
        'Misclass']
    missed_summary_merge['Percentage of Misclassifying'] = missed_summary_merge['Misclass'] / missed_summary_merge[
        'Total Between Two Classes']

    return misclass, correct, missed_summary_merge

def majority_vs_other(results_confusion_matrix, class1, class2):
    # Filter can Start Here
    df_filter_result = results_confusion_matrix[(results_confusion_matrix['True'] == class2) &
                                                (results_confusion_matrix['Predicted'] == class1)]

    # display(df_filter_result)
    # print(df_filter_result.shape[0])

    # Distribution of Locus from Misclassification
    total_miss = results_confusion_matrix['Locus'][
        results_confusion_matrix['Locus'].isin(df_filter_result['Locus'].unique())].value_counts()\
        .reset_index().rename(columns={"Locus": "Total"})

    misses = df_filter_result['Locus']. \
                 value_counts() / results_confusion_matrix['Locus'][results_confusion_matrix['Locus']
        .isin(df_filter_result['Locus'].unique())].value_counts()
    misses = misses.reset_index().merge(total_miss, on='index').rename(columns={"Locus": "Percentage"}).rename(
        columns={"index": "Locus"})

    misses['Value'] = misses["Percentage"] * misses['Total']

    return df_filter_result.shape[0], misses

def mat_analysis(mat, cols):
    total_data = np.sum(mat, axis=1)
    # cols = np.unique(y_true)
    cols_total = np.append(cols, "Total").tolist()
    np.append(total_data / np.sum(total_data), np.sum(total_data))
    df_total = pd.DataFrame([np.append(total_data / np.sum(total_data),
                                       np.sum(total_data))], columns=cols_total)
    df_total['majority'] = df_total.iloc[:, 1:3].idxmax(axis=1)

    return (df_total)
    # df_total.to_csv('../datasets/df_total.csv')
