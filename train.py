from model.model import RedshiftClassifierResNet, RedshiftClassifierInception
from model.eval_model import redshift_evaluate
import numpy as np

data = np.load('data/SDSS-Full/sdss-data-50000.npz', 'r')
X = data['cube']
y = data['labels'][:]['z']
data.close()

train_X, test_X = np.split(X, [11883]) #Should result in 10100 training samples
train_y, test_y = np.split(y, [11883])

print(train_X.shape, test_X.shape)
print(train_y.shape, test_y.shape)

model = RedshiftClassifierInception((64,64,5), max_val=max(train_y, test_y))
model.train(train_X, train_y)

print(model.evaluate())
