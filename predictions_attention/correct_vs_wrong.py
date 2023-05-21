import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
import random

# Add every font at the specified location
font_dir = ['./Siyamrupali.ttf',]
from matplotlib.font_manager import fontManager, FontProperties
path = "./Kalpurush.ttf"
fontManager.addfont(path)
prop = FontProperties(fname=path)
sns.set(font=prop.get_name())
import csv
att_dict = {} 
with open('predictions_attention.csv', encoding="utf8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['ground_truth'] not in att_dict:
            att_dict[row['ground_truth']] = [row['english_word'],row['predicted_word']]
        else:
            del att_dict[row['ground_truth']]

# print(att_dict)
new = []
for key, val in att_dict.items():
    new.append([key, val])

print(len(new))
att_dict = {}
for i in range(10):
    j = random.randint(1, 2475)
    att_dict[new[j][0]] = new[j][1]

print(att_dict)

# Prepare table
columns = ('english_word','ground_truth','predicted_word')

cell_text = []
for gt in att_dict.keys():
    cell_text.append([att_dict[gt][0], gt, att_dict[gt][1]])

# Add a table at the bottom of the axes
# colors = [["w","w","#F0A2A4"],[ "w","w", "#B2E0B1"]]
colors = []
for item in cell_text:
    if item[1] == item[2]:
        colors.append([ "w","w", "#B2E0B1"])
    else:
        colors.append(["w","w","#F0A2A4"])

fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=cell_text,cellColours=colors,
                     colLabels=columns,loc='center')

plt.show()