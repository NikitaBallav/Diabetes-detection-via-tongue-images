# Diabetes-detection-via-tongue-images

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Project Objective:**

The main objective of this research project is to develop a computer-aided decision-making system for the early diagnosis of diabetes using Traditional Chinese Medicine-based analysis of tongue photos. The goal is to provide a tool that assists healthcare professionals in promptly diagnosing diabetes, enabling timely treatment to prevent complications associated with elevated blood glucose levels.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Dataset Description:**

The dataset used in this project comprises tongue photos, from which various properties such as color, texture, shape, dental patterns, and fur are extracted. This dataset consists of 110 images, with 76 images designated for training, 21 for validation, and 13 for testing. These images were collected from an open-source computer vision project website and are selected based on features that are easily obtainable over time and are relevant to diabetes pathology.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Methodology:**

1. **Feature Extraction Using CNNs:** Convolutional Neural Networks (CNNs) are employed to extract relevant features from the tongue photos. These features encompass color, texture, shape, dental patterns, and fur, all of which are essential for diabetes diagnosis from a Traditional Chinese Medicine perspective.

2. **Classification Using Autoencoder Learning:** A classification approach based on autoencoder learning is applied to the extracted features. This method aids in categorizing the tongue photos into diabetic or non-diabetic cases based on the identified properties.

3. **Model Improvement:** The study acknowledges the potential for further research enhancements to improve the effectiveness of the CNN-based machine learning classifier model. Future work may focus on refining and optimizing the model for enhanced diagnostic accuracy.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Results:**

The research project achieved the following outcomes:

- A dataset of 110 tongue photos was utilized, with 76 images for training, 21 for validation, and 13 for testing.

- A binary classifier was developed and applied to the dataset, marking the first-time utilization of this approach for diabetes diagnosis.

- The classifier's performance was evaluated and validated on the test dataset, providing insights into its effectiveness in diagnosing diabetes based on tongue photo properties.

- The project lays the foundation for building a reliable predictor model for diabetes, utilizing a subset of easily accessible features associated with the disease, ultimately reducing the need for extensive patient questioning or testing. Further research and improvements are identified as avenues for future work in this field.
