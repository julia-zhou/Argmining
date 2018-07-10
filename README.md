# Argmining
This repository contains work for our Final Project of TTIC 31190: NLP. It is a collaborative project with Benjamin Rothschild(@b-nroths).

The goal is to explore a method that exploits structured parsing information to detect claims without resorting to topic specific information.

## Abstract
Claim stance classification has become an active field of research in the past years. We show that claim identification could be further improved with features includ- ing mean embedding representation of a sentence, the similarity between sentence and topic target, the location of the sen- tence, length of sentence and PageRank score of the sentence. We also achieve a better than average accuracy instance classification using sentiment analysis be- tween claim and topic. This paper also shows how the combination of many fea- tures types and optimization methods are necessary to achieve accuracy in this NLP task.

## Data
The dataset we are using is the Claim Stance Dataset from IBM Debater project which can be accessed here http: //www.research.ibm.com/haifa/ dept/vst/debating_data.shtml. It contains 2,394 labeled claims for 55 topics that are pulled from 1,065 Wikipedia articles. For each article we are given the following data points:
• Full text from Wikipedia (text) • Topic Target (text)
• Claim Text (text)
• Claim Start Index (integer)
• Claim End Index (integer)
• Stance (Pro or Con)

## Method Overview
We divided this research problem into two parts, claim identification and stance identification.
### 1. Claim Identification
Claim identification is to classify whether a sentence is a claim or not.

### 2. Claim Stance Classification
Stance classification focuses on the problem of de- termining if a claim is PRO or CON towards its topic.

## Conclusion
In this paper, we’ve shown that claim identifica- tion, and claim stance classification could bene- fit from a combination of many features types and optimization methods. A remaining challenge is to build models that are less dependent on train- ing data type and achieve similar results on a vari- ety of data sets including online forum discussions and newspaper articles.

For further description, please check our report [here]:https://github.com/julia-zhou/Argmining/blob/master/paper/Argmining_report.pdf
