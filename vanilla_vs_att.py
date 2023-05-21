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
vanilla_dict = {} 
with open('predictions_vanilla.csv', encoding="utf8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['ground_truth'] not in vanilla_dict:
            vanilla_dict[row['ground_truth']] = row['predicted_word']
        else:
            del vanilla_dict[row['ground_truth']]
compare_list = []
for gt, [eng, indic_pred] in att_dict.items():
    # print(key, value, vanilla_dict[key])
    if gt == indic_pred and vanilla_dict[gt] != indic_pred:
        compare_list.append({'english_word': eng, 'ground_truth': gt, 'attention_prediction': indic_pred, 'vanilla_prediction':vanilla_dict[gt]})

# print(compare_list)
fields = ["english_word", "ground_truth", "attention_prediction", "vanilla_prediction"]

with open('compare_vanilla_vs_att.csv', 'w', encoding="utf8", newline='') as file: 
    writer = csv.DictWriter(file, fieldnames = fields)
    writer.writeheader()
    writer.writerows(compare_list)