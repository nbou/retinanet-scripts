from sklearn import metrics
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
import numpy as np

fle = '/home/nader/scratch/huon_13_GT_SCORE.csv'
my_data = genfromtxt(fle, delimiter=',')
y = my_data[:,0]
scores = my_data[:,1]

fpr, tpr, th = metrics.roc_curve(y, scores)
# print(fpr,tpr)
AUC = metrics.roc_auc_score(y, scores)
plt.figure()
plt.plot(fpr,tpr, color='darkorange', label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


dist = np.sqrt(2)
pt = np.array([0,1])
opt_thresh = 0
opt_tpr = 0
opt_fpr = 0
for i in range(len(th)):
    d = np.linalg.norm(pt-np.array([fpr[i],tpr[i]]))
    if d < dist:
        dist = d
        opt_thresh = th[i]
        opt_fpr = fpr[i]
        opt_tpr = tpr[i]

print('Optimal threshold is {0}'.format(opt_thresh))
print('Opt_fpr: {}'.format(opt_fpr))
print('Opt_tpr: {}'.format(opt_tpr))

precision, recall, thresholds = metrics.precision_recall_curve(y, scores)
AUC_pr = metrics.auc(recall,precision)

plt.plot(thresholds, precision[:-1], 'g--', label='precision')
plt.plot(thresholds, recall[:-1], 'b--', label='recall')
plt.xlabel('Threshold')
plt.legend(loc='lower left')
plt.ylim([0,1])
plt.show()

dist = np.sqrt(2)
pt = np.array([1,1])
opt_thresh = 0
opt_precis = 0
opt_recall = 0
for i in range(len(thresholds)):
    d = np.linalg.norm(pt-np.array([recall[i],precision[i]]))
    if d < dist:
        dist = d
        opt_thresh = thresholds[i]
        opt_recall = recall[i]
        opt_precis = precision[i]

print('Optimal threshold is {0}'.format(opt_thresh))
average_precision = metrics.average_precision_score(y, scores)
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}, AUC:{0:0.2f}'.format(average_precision, AUC_pr))
plt.scatter(opt_recall,opt_precis)
plt.show()