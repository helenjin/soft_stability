for DATASET in tweeteval_emoji tweeteval_emotion tweeteval_hate tweeteval_irony tweeteval_offensive tweeteval_sentiment
do
    for EXP in lime shap intgrad mfaba random
    do
        python3 generate_soft_stability.py \
            --model_name roberta \
            --dataset_name $DATASET \
            --explanation_name $EXP
    done
done

