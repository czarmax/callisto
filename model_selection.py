from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pandas as pd
import matplotlib.pyplot as plt
import operator
import sys
import xlwings as xw
import xlwings.constants
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, qApp, QTextEdit, \
    QPushButton, QHBoxLayout, QWidget, QVBoxLayout, QToolTip, QLineEdit, QLabel, \
    QCheckBox, QComboBox, QGridLayout, QMessageBox, QFileDialog,QFileDialog, \
    QSizePolicy
import webbrowser as web
import clipboard




class BestLinearSearch(LinearRegression, Ridge, Lasso):

    def __init__(self, X, y):
        super().__init__()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
        self.search_ridge()
        self.search_lasso()
        self.learn(X, y)

    def show_plots(self):
        plt.plot(self.model_lin_reg.coef_, "^", label="Linear regression")
        plt.plot(self.model_ridge.coef_, "o", label="Ridge "+str(self.rdg_alpha))
        plt.plot(self.model_lasso.coef_, "*", label="Lasso "+str(self.lasso_alpha))
        plt.xlabel("Coefficient index")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.savefig('coef')
        # plt.show()

    def learn(self, X, y):
        self.model_lin_reg = LinearRegression().fit(X, y)
        self.model_ridge = Ridge(alpha=self.rdg_alpha).fit(X, y)
        self.model_lasso = Lasso(alpha=self.lasso_alpha).fit(X, y)

    def search_ridge(self):  # L2 regularization
        ridge_train_prec = {}
        for alpha in self.alpha_list(0.05, 2):
            model = Ridge(alpha).fit(self.X_train, self.y_train)
            ridge_train_prec[alpha] = model.score(self.X_test, self.y_test)
        self.rdg_list = ridge_train_prec
        self.rdg_alpha = max(ridge_train_prec.items(), key=operator.itemgetter(1))[0]

    def search_lasso(self):  # L1 regularization
        lasso_train_prec = {}
        for alpha in self.alpha_list(0.05, 1):
            model = Lasso(alpha).fit(self.X_train, self.y_train)
            lasso_train_prec[alpha] = model.score(self.X_test, self.y_test)
        self.lasso_list = lasso_train_prec
        self.lasso_alpha = max(lasso_train_prec.items(), key=operator.itemgetter(1))[0]

    @staticmethod
    def alpha_list(granularity, max_value):
        return [i*granularity for i in range(1, int((max_value+1)/granularity))]


class TreeDecision(DecisionTreeClassifier):

    def __init__(self, data, target):  # waiting for DataFrame as data
        super().__init__()
        # selecting exogenous end endogenous variables
        data_X = data.loc[:, data.columns != target]  # endogenous
        data_y = data[target]  # exogenous
        data_dummy = pd.get_dummies(data_X)  # categorization
        self.feature_names = list(data_dummy.keys())  # saving feature names
        data_dummy = data_dummy.values  # converting to numpy array
        data_y = data_y.values  # converting to numpy array
        X, Xt, y, yt = train_test_split(data_dummy, data_y)  # splitting
        test_model = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
        self.test_score = test_model.score(Xt, yt)
        self.model = DecisionTreeClassifier(max_depth=5, random_state=0).fit(data_dummy, data_y)

        #  ATTENTION WE DON T TAKE INTO CONSIDERATION TEST PART


class App(QMainWindow):  # Main application window

    def __init__(self):  # parent=None): # here i don t understand why we are talking about a parent
        super().__init__()  # parent) #we call an init if QMainWindow
        self.start_app()

    def start_app(self):
        hire = Widget()
        self.setCentralWidget(hire)
        self.setFixedSize(700, 500)
        #self.resize(700, 500)
        self.move(300, 300)
        self.setWindowTitle("Callisto")
        self.setWindowIcon(QtGui.QIcon("jupiter.png"))
        self.show()


class Widget(QWidget):  # the widget that is fitted into main window

    def __init__(self):
        super().__init__()
        self.setFixedSize(700, 400)
        self.vbox = QVBoxLayout()
        self.set_file_dialog()
        self.set_learn_process()
        self.set_file_to_predict()
        self.setLayout(self.vbox)

    def set_file_dialog(self):
        # setting learning horizontal layout for file interoperability
        # some of elements are kept outside of current def
        # this is done to be able to interact with them by using
        # outside functions
        head = QLabel('Please, input data for analysis and select learning method.')
        path_lbl = QLabel("Path to data:")
        path = QLineEdit(self)
        path_btn = QPushButton("Select")
        path_btn.clicked.connect(lambda: path.setText(self.choose_directory()))
        proc_btn = QPushButton("Process data")
        proc_btn.clicked.connect(lambda: trgt.addItems(self.process_data(path.text())))
        path_hbox = QHBoxLayout()
        path_hbox.addWidget(path_lbl)
        path_hbox.addWidget(path)
        path_hbox.addWidget(path_btn)
        path_hbox.addWidget(proc_btn)

        # setting target selection
        trgt_lbl = QLabel("Target:")
        trgt = QComboBox()
        trgt.setMinimumSize(600, 25)
        trgt_hbox = QHBoxLayout()
        trgt_hbox.addWidget(trgt_lbl)
        trgt_hbox.addWidget(trgt)

        # setting to vertical layout
        vbox = QVBoxLayout()
        vbox.addWidget(head)
        vbox.addLayout(path_hbox)
        vbox.addLayout(trgt_hbox)
        vbox.addStretch(1)
        self.vbox.addLayout(vbox)
        self.trgt = trgt

    def set_learn_process(self):
        tree_btn = QPushButton('Decision tree')
        tree_btn.setStyleSheet("background-color: green")
        tree_btn.clicked.connect(self.learn_with_tree)

        vis_tree_btn = QPushButton('See tree')
        vis_tree_btn.setToolTip('Open a website and copy the tree to clipboard.')
        vis_tree_btn.clicked.connect(self.visualize_tree)

        self.test_reslt_lbl = QLabel()

        hbox = QHBoxLayout()
        hbox.addWidget(tree_btn)
        hbox.addWidget(vis_tree_btn)
        hbox.addStretch(1)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.test_reslt_lbl)
        hbox2.addStretch(1)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)
        vbox.addStretch(1)
        self.vbox.addLayout(vbox)

    def set_file_to_predict(self):
        # setting learning horizontal layout for file interoperability
        # some of elements are kept outside of current def
        # this is done to be able to interact with them by using
        # outside functions
        head = QLabel('Please, input data for prediction.')
        path_lbl = QLabel("Path to data:")
        path = QLineEdit(self)
        path_btn = QPushButton("Select")
        path_btn.clicked.connect(lambda: path.setText(self.choose_directory()))
        proc_btn = QPushButton("Process data")
        proc_btn.clicked.connect(lambda: trgt.addItems(self.predict_integration(path.text())))
        path_hbox = QHBoxLayout()
        path_hbox.addWidget(path_lbl)
        path_hbox.addWidget(path)
        path_hbox.addWidget(path_btn)
        path_hbox.addWidget(proc_btn)

        # setting target selection
        trgt_lbl = QLabel("Target:")
        trgt = QComboBox()
        trgt.setMinimumSize(600, 25)
        trgt_hbox = QHBoxLayout()
        trgt_hbox.addWidget(trgt_lbl)
        trgt_hbox.addWidget(trgt)

        predict_btn = QPushButton('Predict')
        predict_btn.clicked.connect(self.predict)

        # setting to vertical layout
        vbox = QVBoxLayout()
        vbox.addWidget(head)
        vbox.addLayout(path_hbox)
        vbox.addLayout(trgt_hbox)
        vbox.addWidget(predict_btn)
        vbox.addStretch(1)
        self.vbox.addLayout(vbox)

    def choose_directory(self):
        # we choose index 0 as the following function sends us back
        # a tuple with a path and file type
        directory = QFileDialog.getOpenFileName(self, 'Select directory')[0]
        return str(directory)

    def process_data(self, path):
        # here we integrate data into the program
        # and set list for combobox
        wb = xw.Book(path)
        self.data = wb.sheets(1).used_range.options(pd.DataFrame).value
        #app = xw.apps.active
        #wb.close()
        #app.quit()
        choice_list = []
        for v in self.data.keys():
            choice_list.append(v)
        return choice_list

    def learn_with_tree(self):
        try:
            self.model = TreeDecision(self.data, str(self.trgt.currentText()))
            self.test_reslt_lbl.setText('Test precision result:' + str(self.model.test_score))
            export_graphviz(self.model.model, out_file="tree.dot", feature_names=self.model.feature_names,
                            filled=True)
        except:
            self.adress_learn_error()
        # export_graphviz(model, out_file="tree.dot")  # , class_names=["malignant", "bening"],
        # feature_names=data.feature_names, impurity=False, filled=True)

    @staticmethod
    def visualize_tree(self):
        with open("tree.dot") as f:
            content = f.read()
            clipboard.copy(content)
        web.open('http://www.webgraphviz.com/')

    def predict_integration(self, path):
        # here we integrate data into the program
        self.wb = xw.Book(path)
        return ['Integration completed']

    def predict(self):
        try:
            data = self.wb.sheets(1).used_range.options(pd.DataFrame).value
            cat_data = pd.get_dummies(data).values  # converting to dummies
            result = self.model.model.predict(cat_data)  #model.model!!!!! shity code
            col = last_col(self.wb) + 1
            xw.Range((1, col)).value = 'ALERT PREDICTION'
            for v, k in zip(result, range(2, len(result)+2)):
                xw.Range((k, col)).value = v
        except:
            text = 'Data isn t good looking.'
            message = QMessageBox()
            message.setText(text)
            message.setWindowTitle('Error')


    '''error management in case if the no data selected
    or data format isn't reliable'''

    def adress_learn_error(self):
        text = 'Please select correct path and process data.'
        message = QMessageBox()
        message.setText(text)
        message.setWindowTitle('Error')
        message.setIcon(QMessageBox.Information)
        message.setStandardButtons(QMessageBox.Ok)
        message.exec()

def last_col(wb):
    s = wb.sheets(1)
    RR = s.api.Cells.Find(What="*",
                          After=s.api.Cells(1, 1),
                          LookAt=xw.constants.LookAt.xlPart,
                          LookIn=xw.constants.FindLookIn.xlFormulas,
                          SearchOrder=xw.constants.SearchOrder.xlByColumns,
                          SearchDirection=xw.constants.SearchDirection.xlPrevious,
                          MatchCase=False)
    return RR.Column

if __name__ == "__main__":
    mon_app = QApplication(sys.argv)
    w = App()
    sys.exit(mon_app.exec_())










