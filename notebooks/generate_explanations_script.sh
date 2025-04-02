# for MODEL in vit resnet50 resnet18
# do
#     for EXP in lime shap intgrad mfaba random
#     do
#         python3 generate_explanations.py \
#             --model_name $MODEL \
#             --dataset_name imagenet_2_per_class \
#             --explanation_name $EXP
#     done
# done

for EXP in lime shap intgrad
do
    python3 generate_explanations.py \
        --model_name roberta \
        --dataset_name tweeteval_emotion \
        --explanation_name $EXP
done

