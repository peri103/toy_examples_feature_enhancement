# Overview
This collection of Python scripts is designed to implement and enhance Support Vector Machine (SVM) algorithms for both classification (SVC) and regression (SVR).   The core functionality is encapsulated in the BaseSVM class, which serves as a base class for specific SVM types.

# Files Description
base_svm.py: Defines the BaseSVM class, a foundational class for SVM-based classifiers and regressors.   It handles the basic operations common to all SVM types.
svc.py: Contains the SVC class, derived from BaseSVM, specialized for Support Vector Classification.
svr.py: Contains the SVR class, also derived from BaseSVM, used for Support Vector Regression.

# Edit Intention
The primary intention behind these edits was to incorporate a data scaling feature directly within the SVM classes.   This feature is essential for SVM models as they are sensitive to the scale of input data.   The scale parameter simplifies the process of data preprocessing by automating the scaling within the model, thereby enhancing usability and efficiency.
