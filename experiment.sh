for dataset in dstc11 banking77 #clinc150 hwu64 massive pathic_knowledge
do
    for language_model in paraphrase-multilingual-mpnet-base-v2 sentence-transformers_all-mpnet-base-v2 #bert-base-multilingual-cased xlm-roberta-base
    do
        for loss in triplet_plus_supervised #triplet_margin_loss supervised_learning triplet_plus_supervised binary_classification contrastive_learning cosine_similarity_loss
        do
            for round in 0 1 2 3 4
            do
                PYTHONPATH=. python3 ./src/train.py dataset_specific_hyperparams=$dataset model.base_model_name=$language_model model.training_objective=$loss model.C=15 model.r=0.5 model.triplet_margin=0.15 model.contrastive_margin=1.75 data.experiment_round=$round
            done
        done
    done
done

