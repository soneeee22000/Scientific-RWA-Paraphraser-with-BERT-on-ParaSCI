# Paraphraser Bert Model Training On ParaSCI Dataset

This repo contains my contribution to the Paraphrasing Project at AIT Brain-Lab with the BERT Model on **ParaSCI** dataset
in [**ParaSCI: A Large Scientific Paraphrase Dataset for Longer Paraphrase Generation**](https://arxiv.org/abs/2101.08382) along with the code working on tasks ranging from ETL- Extract,Transform,Load , some EDA analysis(always) ,preprocessing, the creating training sequence to the actual training , fine tuning , deployment and so forth...

# Introduction

Recap on " What it means to Paraphrase something"A paraphrase is a restatement of meaning with different expressions. Being very common in our daily language expressions, it can also be applied to multiple downstream tasks of NLP, such as generating diverse text or adding richness to a chatbot.

When it comes to choosing our Datasets for both pretraining and Finetuning , we propose ParaSCI, the first large-scale paraphrase dataset in the scientific field, including 33,981 paraphrase pairs from ACL (ParaSCI-ACL) and 316,063 pairs from arXiv (ParaSCI-arXiv). [GitHub - allenai/s2orc: S2ORC: The Semantic Scholar Open Research Corpus: https://www.aclweb.org/anthology/2020.acl-main.447/](https://github.com/allenai/s2orc)

## Download the Dataset

We are going to download Reformatted version of the ParaSCI dataset from [ParaSCI: A Large Scientific Paraphrase Dataset for Longer Paraphrase Generation](https://arxiv.org/abs/2101.08382). Data retrieved from [dqxiu/ParaSCI](https://github.com/dqxiu/ParaSCI) from HuggingFace Datasests !

According to HuggingFace ,we can do that easily by [Load a dataset from the Hub (huggingface.co)](https://huggingface.co/docs/datasets/load_hub)

Since it's reformatted , we need to relearn all its properties through the EDA process and it can be learned as follows:

### Detailed Statistics for ParaSCI 

· Len represents the average number of words per sentence and Char Len represents the average number of characters per sentence.

· We calculate Len, Char Len and Self-BLEU of the reformatted dataset from Huggingface mostly made up of gold-standard paraphrases

| Name          | Source |  | Target | Len          | Char Len      | Self-BLEU     |
| ------------- | ------ | - | ------ | ------------ | ------------- | ------------- |
| ParaSCI train | 338717 |  | 338717 | 18.849133642 | 115.715352344 | 0.24567297508 |
|               |        |  |        |              |               |               |

![image](https://github.com/soneeee22000/Scientific-RWA-Paraphraser-with-BERT-on-ParaSCI/assets/109932809/bf806142-edb1-4f68-a3cf-ca920fe230be)


## Our Model

Since In the case of our BERT model (Bidirectional Encoder Representations from Transformers), a small portion of the input sequence is randomly masked(during Pretraining), meaning certain words in our train sequence are replaced with a special token, typically "[MASK]". The model then tries to predict the original words based on the context of the surrounding words.

By learning to predict masked words, BERT gains a deeper understanding of the relationships between words in a sentence. It captures both the left and right contexts of a word, allowing it to capture rich contextual information.

After pre-training, the BERT model can be fine-tuned on specific downstream tasks, such as text classification or named entity recognition. The pre-training on masked language modeling helps the model to better generalize and understand the context in the downstream tasks.

---
