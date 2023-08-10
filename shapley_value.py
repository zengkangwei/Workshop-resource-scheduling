#########################################################
### MULTI-TOUCH ATTRIBUTION MODEL USING SHAPLEY VALUE ###
#########################################################
# %%
### import packages
import itertools
from pathlib import Path
from collections import defaultdict
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
### reading in dataset
data_path = './dataset/datasets1.csv'
data_path_resolve = Path(data_path).resolve()
data_raw = pd.read_csv(data_path_resolve)

# %%
###  extracting the needed field
columns = ['marketing_CP_subset', 'payback']
data = data_raw[columns].copy()
print(f"Data dimension: {data.shape}")

# %%
### create a channel mix conversion table
# first level - sort
data_lvl1 = data[['marketing_CP_subset', 'payback']]
# second level - groupby userid, concat distinct marketing channel and label if any conversion took place with this channel mix
print(data_lvl1)

# %%
#### setting up the formulas for shapley value
#######################################################################################################################

def power_set(List):
    PS = [list(j) for i in range(len(List)) for j in itertools.combinations(List, i + 1)]
    return PS


#######################################################################################################################

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


#######################################################################################################################

def v_function(A, C_values):
    '''
    This function computes the worth of each coalition.
    inputs:
            - A : a coalition of channels.
            - C_values : A dictionnary containing the number of conversions that each subset of channels has yielded.
    '''
    subsets_of_A = subsets(A)
    worth_of_A = 0
    for subset in subsets_of_A:
        if subset in C_values:
            worth_of_A += C_values[subset]
    # print(worth_of_A)
    # print(".....................")
    return worth_of_A


#######################################################################################################################

def subsets(s):
    '''
    This function returns all the possible subsets of a set of channels.
    input :
            - s: a set of channels.
    '''
    if len(s) == 1:
        return s
    else:
        sub_channels = []
        for i in range(1, len(s) + 1):
            sub_channels.extend(map(list, itertools.combinations(s, i)))
    # print(list(map(",".join, map(sorted, sub_channels))))
    # print("........................")
    return list(map(",".join, map(sorted, sub_channels)))

#######################################################################################################################

def price(s):
    '''
    This function returns all the possible subsets of a set of channels.
    input :
            - s: a set of channels.
    '''
    if len(s)==1:
        return s
    else:
        sub_channels=[]
        for i in range(1,len(s)+1):
            sub_channels.extend(map(list,itertools.combinations(s, i)))
    # print(list(map(",".join, map(sorted, sub_channels))))
    # print("........................")
    return list(map(",".join,map(sorted,sub_channels)))
#######################################################################################################################

def calculate_shapley(df, cp_name, payback_name):
    '''
    This function returns the shapley values
            - df: A dataframe with the two columns: [cp_name, payback_name].
            The channel_subset column is the channel(s) associated with the conversion and the count is the sum of the conversions.
            - cp_name: A string that is the name of the cp column
            - conv_name: A string that is the name of the column with conversions
            **Make sure that that each value in channel_subset is in alphabetical order. Email,PPC and PPC,Email are the same
            in regards to this analysis and should be combined under Email,PPC.

    '''
    # casting the subset into dict, and getting the unique channels
    c_values = df.set_index(cp_name).to_dict()[payback_name]
    alliances = df[cp_name].apply(lambda x: x.split(",")).explode().unique()

    # v_values = {}
    # for A in power_set(alliances):  # generate all possible channel combination
    #     v_values[','.join(sorted(A))] = v_function(A, c_values)
    #     print(v_values)
    n = len(alliances)  # no. of channels
    shapley_values = defaultdict(int)

    for channel in alliances:
        for A in c_values.keys():
            if channel not in A.split(","):
                cardinal_A = len(A.split(","))
                A_with_channel = A.split(",")
                A_with_channel.append(channel)
                A_with_channel = ",".join(sorted(A_with_channel))
                weight = (factorial(cardinal_A) * factorial(n - cardinal_A - 1) / factorial(
                    n))  # Weight = |S|!(n-|S|-1)!/n!
                contrib = (c_values[A_with_channel] - c_values[A])  # Marginal contribution = v(S U {i})-v(S)
                shapley_values[channel] += weight * contrib
        # Add the term corresponding to the empty set
        shapley_values[channel] += c_values[channel] / n

    return shapley_values


# %%
### calculating the shapley value of the channel
shapley_dict = calculate_shapley(data_lvl1, 'marketing_CP_subset', 'payback')
shapley_result = pd.DataFrame(list(shapley_dict.items()), columns=['CP', 'shapley_value'])
print(shapley_result)
# %%
### visualizing the results
sns.set_style("white")

plt.subplots(figsize=(15,8))
s = sns.barplot(x='CP', y='shapley_value', data=shapley_result)

for idx, row in shapley_result.iterrows():
    s.text(row.name, row.shapley_value +6, round(row.shapley_value,1), ha='center', color='darkslategray', fontweight='semibold')
plt.title("CP'S SHAPLEY VALUE",
          fontdict={'fontfamily': 'san-serif', 'fontsize': 15, 'fontweight': 'semibold', 'color':'#444444'},
          loc='center', pad=10)
plt.show()


# %%
