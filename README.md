# Overview

Train a domain classifier for IT and Computer Science using abstracts as input. The trained model can then be used to classify abstracts and awards from various sources (where categories and labels are not available), allowing for better analysis across the various data sources being collected as part of the the project.

## Thoughts and discussion
* Different CS categories available in WoS might not be of much value and classification accuracy around 60% currently -  might want to consider binary classification between "IT or CS related" and "Other"
	* Current classification categories:
		* Computer Science, Artificial Intelligence: 2502
		* Computer Science, Cybernetics: 749
		* Computer Science, Hardware & Architecture: 2393
		* Computer Science, Information Systems: 2011
		* Computer Science, Interdisciplinary Applications: 1623
		* Computer Science, Software Engineering: 847
		* Computer Science, Theory & Methods: 1529
		* Other: lots available
* For many of the data sources without domain labels (e.g. NSF, DOE), there are some sanity checks using other fields/info that can indicate how well classifier is performing.
* Abstracts are often tagged with more than one category but this process currently only considers the primary category for each abstract. This could potentially make it difficult for the classifier to distinguish between classes and it may be wise to train with abstracts only tagged with one category to improve model differentiation. However, in reality many abstracts being tagged will belong to more than one class and we will like want tag multiple categories using softmax output. 

## Web of Science Data

Location: /data/shared/domain-classification/data.zip

Abstracts for various categories from Web of Science provide labeled data (categories) to train classifier.

## NTIS Data

Location: Asitang to add

The NTIS data is also labeled with similar categories and can be used in training and evaluation to help ensure the model generalizes. There is also a lot more data here. A mapping between the WoS labels and NTIS labels will need to be created for apples to apples comparisons and to using both in building classifier. 

## Keywords

Both Web of Science and NTIS have keywords available for abstracts that can be used to tag abstracts without keywords available. Relationships between these keywords can then be used to do more granular characterization of abstracts beyond the classification categories. Current plan is to combine keywords lists from CS-related abstracts from NTIS and WoS data. 