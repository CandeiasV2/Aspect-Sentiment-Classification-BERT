# Sentiment Analysis for Neural Information Processing and Retrieval
This project was accomplished for the "Neural Information Processing and Retrieval" class. The task was to attempt to outperform the baseline model proposed in the paper "[Aspect-based Sentiment Classification via Reinforcement Learning](https://ieeexplore.ieee.org/document/9679112)" using an alternative model on the same dataset.

## Overview
The focus of this project was on aspect-based sentiment classification (ABSC), which involves determining the sentiment of a specific aspect within a given text. Using the Twitter dataset, we aimed to improve sentiment classification accuracy. The approach leveraged a fine-tuned pretrained BERT model from the HuggingFace library, renowned for its capability to handle complex natural language processing (NLP) tasks effectively.

Given the project’s limited time frame, we used a [pretrained BERT model](https://huggingface.co/docs/transformers/model_doc/bert) for its simplicity and efficiency, eliminating the need for extensive data and training. Input sentences were preprocessed by removing non-alphabetical characters, improving the quality of the input data before passing it into the BERT model for sentiment classification.

## Methodology
1. **Data Preprocessing:** The input text was cleaned by removing punctuation, numbers, and other non-alphabetic characters. This preprocessing step enhanced the accuracy of the model by eliminating noise that could affect sentiment analysis.

2. **Model Selection:** After considering various models introduced in class, the BERT model was chosen due to its proven effectiveness in handling NLP tasks and its availability as a pretrained model. This choice allowed for fine-tuning on the Twitter dataset with relatively minimal training time.

3. **Implementation:**
    - Used the HuggingFace Transformers library to fine-tune the BERT model.
    - Concatenated input sentences with the target aspect using a separator token ([SEP]) to create a suitable input for the BERT model.
    - Defined the model using PyTorch, incorporating a dropout layer and a linear layer for multi-label classification (negative, neutral, positive).
    - The model was trained with a batch size of 8, a learning rate of 1e-5, and the 'BCEWithLogitsLoss()' function for optimal performance.

4. **Training and Evaluation:**
    - Trained the model on an RTX 3060 TI GPU to expedite the process. Each epoch took approximately 2 minutes to complete.
    - Used accuracy and macro-F1 score as evaluation metrics, the same metrics used in the baseline model, to ensure a fair comparison.

## Results
The proposed model achieved an accuracy of 74.13% and a macro-F1 score of 73.05% on the Twitter dataset. This marked an improvement of 1.41% over the baseline model (SentRL) from the original paper, which recorded an accuracy of 73.12% and a macro-F1 score of 71.64%. This performance demonstrates the effectiveness of the BERT model, even with relatively simple preprocessing and fine-tuning.

<div align="center">

| Methods     | Accuracy | Macro-F1 |
|-------------|----------|----------|
| SVM [1]     | 63.40    | 63.30    |
| LSTM [1]    | 69.56    | 67.70    |
| MenNet [1]  | 71.48    | 69.90    |
| AOA [1]     | 72.30    | 70.20    |
| IAN [1]     | 72.50    | 70.81    |
| TNet-LF [1] | 72.98    | 71.43    |
| ASCNN [1]   | 71.05    | 69.45    |
| ASGCN-DT [1]| 71.53    | 69.68    |
| ASGCN-DG [1]| 72.15    | 70.40    |
| SentRL [1]  | 73.12    | 71.64    |
| **Our Model**   | **74.13**    | **73.05**    |

</div>


## Conclusion
Using a pretrained BERT model with effective text preprocessing, we successfully surpassed the baseline model's performance in aspect-based sentiment classification on the Twitter dataset. Despite some limitations, such as longer training times and sensitivity to learning rates, this approach shows promise for further optimization. Future work could explore lighter-weight versions of BERT, like DistilBERT, to reduce computational costs while maintaining high accuracy.

## References

[1] L. Wang et al., “Aspect-based sentiment classification via reinforcement learning,” 2021 IEEE International Conference on Data Mining (ICDM), pp. 1391–1396, Dec. 2021. [doi:10.1109/icdm51629.2021.00177](https://doi.org/10.1109/icdm51629.2021.00177)

[2] “Bert - Hugging face,” Hugging Face, [https://huggingface.co/docs/transformers/model_doc/bert](https://huggingface.co/docs/transformers/model_doc/bert) (accessed Nov. 15, 2023).

