## Breast Tumor Diagnosis 

### Dataset
```python
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
```

### Idea
The Diagnostic Wisconsin Breast Cancer dataset ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)) contains features extracted from digitized images of fine needle aspirates (FNA) of breast masses to support tumor diagnosis. Each record represents a single cell-sample and includes a diagnosis label (M = malignant, B = benign). The dataset has 569 instances with a class distribution of 212 malignant and 357 benign samples.

For this week, try to implement the perceptron algorithm for prediction.

### Recommendations 
1. dimensionality reduction 
2. select relevant features
3. 80/20 for training and testing

### Final Remark
1. try not to directly use top-level API for implementation
2. Have fun!