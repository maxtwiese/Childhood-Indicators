"""RandomForestClasifier prediciting underage drinking

Builds a recoded and imputed DataFrame from NSDUH_2019 data with
pipeline.py.  Base Model is guessing that all respondents
have experimented with alcohol.  Uses a Random Forest Classifier to
determine if a child will drink focused on recall for evaluation metric.
Trains model with Stratified K-Fold Cross Validation and maximizing
F1-Score by using the Precisions-Recall Cruve to select a threshold.
Then, using Permuation Feature importnace most important indicators are
determined.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, recall_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from rfpimp import permutation_importances
from pipeline import build_vars, imputer, recoder, youth_df
plt.style.use('ggplot')
plt.rcParams.update({'font.size':16})

# Build the DataFrame, set the Base Model, & instantiate Model
df = youth_df()
df = recoder(df)
df = imputer(df)

rf = RandomForestClassifier(n_estimators=1000,
                            max_features='auto',
                            random_state=1618,
                            class_weight='balanced')                           
skf = StratifiedKFold(n_splits=10,
                      shuffle=True,
                      random_state=1618)
A, A_test, d, d_test = build_vars(df, 'ALCOHOL')

no_skill = len(d[d==1])/len(d)
out_of_box_recall = cross_val_score(rf, A, d, scoring='recall', cv=skf).mean()

print(f"\nBase Model\n\tRecall: 1\n\tAccuracy: {no_skill:.3f}")
print(f"Out of Box RF\n\tRecall: {out_of_box_recall:.3f}\n")

# Find best threshold using Precision-Recall Curve to maximize F1-Score
thresholds = []
recalls = []
fig1, ax1 = plt.subplots(figsize = (20, 15))
legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='Max F1', markersize=12),
    Line2D([0],[0], linestyle='--', color = 'blue', label='No Skill')
    ]

for train, val in skf.split(A, d):

    rf.fit(A.values[train], d.values[train])
    d_hat = rf.predict_proba(A.values[val])
    d_hat = d_hat[:,1]
    p, r, th = precision_recall_curve(d.values[val], d_hat)
    f1_score = 2 * p * r / (p + r)
    ix = np.argmax(f1_score)
    ax1.plot(r, p, marker='.', alpha=0.5)
    ax1.plot(r[ix], p[ix], marker='o', color='black', markersize = 12)
    thresholds.append(th[ix])
    recalls.append(r[ix])

ax1.plot([0,1], [no_skill,no_skill], linestyle='--', color='blue',
         label='No Skill')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.legend(handles=legend_elements, loc='upper right')
ax1.set_title('Precision Recall Curve', fontsize=36)
plt.tight_layout()
plt.show()
fig1.savefig(r'../images/PRC')

print('Cross Validated Recall/Treshold for MAX F1 Score')
[print("\tRecall: {:.3f} -- Threshold: {:.3f}".format(recall, threshold)) \
    for recall, threshold in zip(recalls, thresholds)]

# Get training recall with cross validation, fit to data, and test.
thresh = thresholds[np.argmax(recalls)]
recalls = []

for train, val in skf.split(A, d):

    rf.fit(A.values[train], d.values[train])
    d_hat = rf.predict_proba(A.values[val])
    d_hat = (d_hat[:,1] >= thresh).astype('int')
    recalls.append(recall_score(d.values[val], d_hat))

rf.fit(A, d)
d_hat = rf.predict_proba(A_test)
d_hat = (d_hat[:,1] >= thresh).astype('int')

print(f"Training Recall Score of {np.mean(recalls):.3f}.")
print(f"Test Recall Score of {recall_score(d_test, d_hat):.3f}.")
print(f"An improvment of {rf.score(A_test,d_test) - no_skill:.3f} in acccuracy"\
    " from the no skill guess.\n")

# Determine Feature Importance 
def r2(rf, A_train, d_train):
    d_hat = rf.predict_proba(A_train)
    d_hat = (d_hat[:,1] >= thresh).astype('int')
    return r2_score(d_train, d_hat)

perm_imp_rfpimp = permutation_importances(rf, A, d, r2)

ytl = [
'Age', '# Students in your grade that\ndrink Alcohol', 
'How would you feel if someone your\nage used Marijuana monthly?',
'How would you feel if someone your\nage drank daily',
'How would you feel if someone your\nage tried Marijuana?',
'How would you feel if your friend\nused Marijuana monthly?',
'How would you feel if your friend\ntried Marijuana?',
'How often do you fight\nwith your parents?',
'How would your parents feel if you\ntried Marijuana?',
'# Students in your grade that\nsmoke Marijuana'
]

fig, ax = plt.subplots(figsize=(20, 15))
perm_imp_rfpimp[:10].plot.barh(ax=ax, color = 'purple')
ax.set_xlabel('Importance')
ax.set_title('Top Indicators For Consuming Alcohol Age < 18', fontsize=36)
ax.get_legend().remove()
ax.set_ylabel('')
ax.set_yticklabels(ytl)
plt.tight_layout()
fig.savefig(r'../images/Top_Indicators')
plt.show()
