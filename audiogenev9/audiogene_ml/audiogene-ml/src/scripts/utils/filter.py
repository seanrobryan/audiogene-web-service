# Filter Ages
def filter_ages(df, age_group):
    if '-' in age_group:
        ages = age_group.split('-')
        df_new = df.loc[(df.age >= float(ages[0])) & (df.age < float(ages[1]))]
    elif '+' in age_group:
        ages = age_group.split('+')
        df_new = df.loc[(df.age >= float(ages[0]))]
    else:
        df_new = df

    return df_new

def filter_instance_groups(df, instance_group):
    return df[df['instance_group'] == instance_group]

def filter_shapes(df, shape):
    return df[df['shape'] == shape]