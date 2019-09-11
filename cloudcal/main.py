from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

def svm_1():
    from sklearn.svm import SVC
    model=SVC(kernel='linear', C=1)
    return model

def svr():
    from sklearn.svm import SVR
    model =SVR()
    return model

def load_data():
    data=pd.read_csv('')
    return data

def k_fold(data):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=False)
    kf.split(data)

if __name__ == '__main__':
    iris = datasets.load_iris()  #iris数据集作为测试
    train_X, train_Y = iris.data, iris.target
    train_X, test_x, train_Y,test_y = train_test_split(train_X, train_Y, test_size=.3)
    print(train_X.shape, train_Y.shape)
    clf=svr()
    clf.fit(train_X,train_Y)
    predict1=clf.predict(test_x)
    print(predict1)
    gongshi=clf.get_params()
    print(gongshi)

    # clf = svm_1()
    # scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    # scores1=scores.mean()
    # print(scores)
    # print(scores1)
