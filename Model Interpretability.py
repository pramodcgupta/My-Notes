# Model Interpretability

LIME
eli5
shap

# ---------------------------------------------- LIME ---------------------------------------------------------------------------

# Load our pkgs
import lime
import lime.lime_tabular

# Feature Names
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Class Names
class_names = Y.unique()

'''
Explainers For Lime
- Create the explainer
- LimeTabularExplainer : for tabular data/columns
- LimeTextExplainer: for text/words
- LimeImageExplainer: for images
'''

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=class_names, discretize_continuous=True)

# The Explainer Instance
exp = explainer.explain_instance(X_test.iloc[8],model_logreg.predict_proba,num_features=4,top_labels=1)

# Show in notebook
exp.show_in_notebook(show_table=True, show_all=False)

# ------------------------------------------------------ eli5 ------------------------------------------------------------------------

'''
## --------------- Using Eli5
- Eli5 (Explain It Like I am 5)
- ELI5 understands text processing utilities and can highlight text data accordingly. 
- Shows the weight and bias(intercept)
- pip install eli5

# ------------------- Building the Explainer for Eli5
- Provide Model
- Feature Names
- Class names/labels
'''

import eli5

# Clearly Define Feature Names
eli5.show_weights(model_logreg,feature_names=feature_names,target_names = class_names)

# Show Explaination For A Single Prediction
eli5.show_prediction(model_logreg, X_test.iloc[10],feature_names=feature_names,target_names=class_names)

