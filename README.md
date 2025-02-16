# Acne Analysis Using Deep Learning with EfficientNetB0

Excited to share our latest deep learning project on Acne Analysis! 🌟🔍 Leveraging EfficientNetB0, we developed a model to classify facial images into five acne conditions:

- Blackheads
- Cyst
- Papules
- Pustules
- Whiteheads

This system not only identifies these conditions but also provides personalized recommendations for each detected issue, empowering users to better understand their skin health.

## Dataset Overview

- **Total Images:** 4,607 (Train: 2,768 | Validation: 921 | Test: 918)
- **Preprocessing:** Image resizing (150x150), normalization, augmentation
- **Augmentation Techniques:** Horizontal flipping, rotation, zooming, brightness adjustment, contrast variation, Gaussian noise

## Model Training & Implementation

- **Backbone Architecture:** EfficientNetB0 (pre-trained on ImageNet)
- **Custom Layers:** Dense layers with dropout (up to 60%), L2 regularization, and batch normalization

### Optimized Training

- **Batch Size:** 32
- **Optimizer:** Adam with AMSGrad (learning rate = 0.0001)
- **Loss Function:** Categorical Crossentropy
- **Callbacks:** Early Stopping, ReduceLROnPlateau, Class Weights

## Results & Performance

**Final Model Performance:**

- **Test Accuracy:** 96.84%
- **Test Precision:** 97.05%
- **Test Recall:** 96.62%
- **F1-Score (Macro Avg):** 96.74%

### Confusion Matrix Insights

The model achieved near-perfect predictions for Pustules and Whiteheads, with 100% confidence in test cases.

## Insights & Takeaways

- **EfficientNetB0** excelled in extracting fine-grained features from facial images.
- **Data Augmentation** significantly improved generalization despite the small dataset.
- **Class Weights** effectively mitigated class imbalance, ensuring fair representation.

## Project Credits

Big thanks to my amazing teammates for their contributions:
- **Ansh Chauhan:** Model creation and optimization
- **Harshit Saini:** Documentation and testing
- **Tushar Soni:** Testing and deployment

Would love to hear thoughts from the AI and healthcare community! Let’s connect and collaborate to take this project to the next level. 🚀

#DeepLearning #MedicalAI #EfficientNet #SkinHealth #AcneAnalysis #HealthcareAI #MachineLearning #ComputerVision
