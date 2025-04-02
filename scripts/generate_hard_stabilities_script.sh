for DATASET in tweeteval_emoji tweeteval_emotion tweeteval_hate tweeteval_irony tweeteval_offensive tweeteval_sentiment
do
    for EXP in lime shap intgrad mfaba random
    do
        python3 generate_hard_stabilities.py \
            --model_name roberta \
            --dataset_name $DATASET \
            --explanation_name $EXP
    done
done


for MODEL in vit resnet50 resnet18
do
    for EXP in lime shap intgrad mfaba random
    do
        python3 generate_hard_stabilities.py \
            --model_name $MODEL \
            --dataset_name imagenet_2_per_class \
            --explanation_name $EXP
    done
done

