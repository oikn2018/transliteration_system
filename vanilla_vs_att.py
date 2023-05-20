import csv
att_dict = {} 
with open('predictions_attention.csv', encoding="utf8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['ground_truth'] not in att_dict:
            att_dict[row['ground_truth']] = row['predicted_word']
        else:
            del att_dict[row['ground_truth']]

vanilla_dict = {} 
with open('predictions_vanilla.csv', encoding="utf8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['ground_truth'] not in vanilla_dict:
            vanilla_dict[row['ground_truth']] = row['predicted_word']
        else:
            del vanilla_dict[row['ground_truth']]
compare_list = []
for key, value in att_dict.items():
    # print(key, value, vanilla_dict[key])
    if key == value and vanilla_dict[key] != value:
        compare_list.append({'ground_truth': key, 'attention_prediction': value, 'vanilla_prediction':vanilla_dict[key]})

# print(compare_list)
fields = ["ground_truth", "attention_prediction", "vanilla_prediction"]

with open('compare_vanilla_vs_att.csv', 'w', encoding="utf8", newline='') as file: 
    writer = csv.DictWriter(file, fieldnames = fields)
    writer.writeheader()
    writer.writerows(compare_list)