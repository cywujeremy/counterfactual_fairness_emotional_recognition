import os
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# set categorical and attribute tables
table_cat = {"F": defaultdict(int), "M": defaultdict(int)}
table_attr = {"F": defaultdict(list), "M": defaultdict(list)}
# F:{"ang":{"valence": [], "activation": [], "dominance": []}}
table_gender_cat = {"F": {}, "M": {}}

# record all data
for session_num in range(1, 6):
    # get path and files
    path = '../IEMOCAP_full_release/Session' + str(session_num) + '/dialog/EmoEvaluation'
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f[0] != "."]
    logging.info("working on dir: " + path)
    logging.info("total files: " + str(len(files)))

    # record categorical and attribute infor corresponding to gender
    for file_name in files:
        with open(os.path.join(path, file_name), 'r') as fd:
            for turn in fd.read().split("\n\n")[1:-1]:
                head = turn.split("\n")[0].split("\t")
                # ['[6.7700 - 8.4600]', 'Ses01M_impro01_F000', 'ang', '[1.5000, 3.5000, 4.5000]']
                # time | file_name | categorical | attribute
                gender = head[1][-4]

                # record categorical info
                cat = head[2]
                table_cat[gender][cat] += 1

                # record attribute info
                attr = head[3][1:-1].split(", ")
                table_attr[gender]["valence"].append(float(attr[0]))
                table_attr[gender]["activation"].append(float(attr[1]))
                table_attr[gender]["dominance"].append(float(attr[2]))

                # record emotion attribute by gender
                if cat not in table_gender_cat[gender]:
                    table_gender_cat[gender][cat] = {"valence": [], "activation": [], "dominance": []}

                table_gender_cat[gender][cat]["valence"].append(float(attr[0]))
                table_gender_cat[gender][cat]["activation"].append(float(attr[1]))
                table_gender_cat[gender][cat]["dominance"].append(float(attr[2]))

    logging.info("finish...")

total_F = 0
total_M = 0
for key in table_cat["M"]:
    total_M += table_cat["M"][key]

for key in table_cat["F"]:
    total_F += table_cat["F"][key]

print(total_M)
print(total_F)
# plot four charts
plt.rcParams["figure.figsize"] = [7.50, 7.5]
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, gridspec_kw={'height_ratios': [4,1,1,1]})

# plot categorical chart
labels = table_cat["M"].keys()
x = np.arange(len(labels))
width = 0.35

male_result = []
female_result = []

for emo in labels:
    male_result.append(table_cat["M"][emo])
    female_result.append(table_cat["F"][emo])

rects1 = ax1.bar(x - width/2, female_result, width, label='Female')
rects2 = ax1.bar(x + width/2, male_result, width, label='Male')

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel('Count')
ax1.set_title('EmoEvaluation result by gender (categorical)')
ax1.legend()

# plot attribute analysis
stat = [np.mean, np.std, np.var]
stat_labels = ["Mean", "Standard Deviation", "Variance"]
axes = ["valence", "activation", "dominance"]
width_attr = 0.2
for _index, (axe, ax) in enumerate(zip(axes, [ax2, ax3, ax4])):
    male_result = []
    female_result = []

    # compute statistic info
    for _func in stat:
        male_result.append(_func(table_attr["M"][axe]))
        female_result.append(_func(table_attr["F"][axe]))

    print(male_result)
    print(female_result)
    # plot chart
    x = np.arange(len(stat_labels))
    rects1 = ax.bar(x - width_attr / 2, female_result, width_attr, label='Female')
    rects2 = ax.bar(x + width_attr / 2, male_result, width_attr, label='Male')
    ax.set_xticks(x)
    ax.set_xticklabels(stat_labels)
    ax.set_title('EmoEvaluation result by gender (attribute - ' + axes[_index] + ")")

# plot all chars
fig.tight_layout()
plt.savefig("dataset_analysis")

with open('table_gender_cat.csv', 'w') as f:
    f.write("Gender,Category,Number,Attribute,Mean,Standard_Deviation,Variance\n")
    for gender in table_gender_cat.keys():
        for cat in table_gender_cat[gender]:
            result = []
            for attr in table_gender_cat[gender][cat]:
                result = []
                num = []
                for _index, func in enumerate(stat):
                    result.append(func(table_gender_cat[gender][cat][attr]))
                    num.append(len(table_gender_cat[gender][cat][attr]))
                assert num[0] == num[1] == num[2]
                f.write("%s,%s,%s,%s,%s,%s,%s\n" % (gender, cat, num[0], attr ,result[0], result[1], result[2]))



