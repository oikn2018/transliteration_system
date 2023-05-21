import wandb
wandb.init(project="CS6910_Assignment3", entity="dl_research", name="Question 4")

 
# plt.imshow(fig)
image = f"vanilla_pred.png"
label = 'Predictions by Vanilla Seq2Seq [Green: Correct Predictions, Red: Incorrect Predictions]'
# image_list.append(image)
# label_list.append(classify)

wandb.log(wandb.Image(image, caption=label))