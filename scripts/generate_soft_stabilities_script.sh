for MODEL in resnet18 resnet50 vit
do
    for EXP in lime shap intgrad mfaba random
    do
        for FRAC in 0.125 0.25 0.375 0.5
        do
            python3 generate_soft_stabilities.py \
                --model_name $MODEL \
                --dataset_name imagenet_2_per_class \
                --explanation_name $EXP \
                --top_k_frac $FRAC
        done
    done
done


for DATASET in tweeteval_emoji tweeteval_emotion tweeteval_hate tweeteval_irony tweeteval_offensive tweeteval_sentiment
do
    for EXP in lime shap intgrad mfaba random
    do
        for FRAC in 0.125 0.25 0.375 0.5
        do
            python3 generate_soft_stabilities.py \
                --model_name roberta \
                --dataset_name $DATASET \
                --explanation_name $EXP \
                --top_k_frac $FRAC
        done
    done
done
