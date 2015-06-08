import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold

from sklearn.svm import SVC


class KDC:
    ONE_HOT_ENCODING = "one_hot_encoding"
    LABEL_ENCODING = "label_encoding"
    UNIVARIATE_FEATURES_SELECTION = "univariate_features_selection"
    RECURSIVE_FEATURE_ELIMINATION = "recursive_feature_elimination"
    DIMENSION_REDUCTION = "dimension_reduction"

    def __init__(self, train_data: str, test_data: str, encoding: str, feature: str):
        # self.train_data = train_data
        # self.test_data = test_data
        self.encoding = encoding
        self.feature_selection = feature
        self.x, self.y = self.data_preparation(train_data, test_data)
        self.features = []
        self.attack_categories = {}

    @staticmethod
    def plot_variance(df, variance_ratio):
        m, n = df.shape
        plt.figure()
        plt.title("Variance plot")
        plt.xlabel("Component number")
        # x plt.xlim(1, n)
        plt.ylabel("Proportion of variance")
        # plt.ylim(0, 1)
        plt.plot(np.arange(1, n + 1), variance_ratio)
        plt.show()

    def data_preparation(self, train_data: str, test_data: str) -> (pd.DataFrame, pd.DataFrame):
        names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                 "wrong_fragment",
                 "urgent",
                 "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
                 "num_root",
                 "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
                 "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
                 "srv_rerror_rate",
                 "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                 "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                 "dst_host_rerror_rate",
                 "dst_host_srv_rerror_rate", "result"]

        train_df = pd.read_csv(train_data, names=names)
        test_df = pd.read_csv(test_data, names=names)
        df = pd.concat([train_df, test_df], ignore_index=True)
        assert isinstance(df, pd.DataFrame)

        # if data is already cleaned, then don't clean it again
        # if not redo and os.path.isfile(train_data + "_reduced.csv"):
        #     x = pd.read_csv(train_data + "_reduced.csv", names=names[:-1])
        #     y = pd.read_csv(train_data + '_Y_cleaned.csv')
        #     return x, y

        ddos = ["back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb", "processtable", "udpstorm",
                "apache2",
                "worm", "probe"]
        u2r = ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"]
        r2l = ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient", "xlock", "xsnoop",
               "snmpguess",
               "snmpgetattack", "httptunnel", "sendmail", "named", "spy"]
        probe = ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]
        self.attack_categories = {"ddos": ddos, "u2r": u2r, "r2l": r2l, "probe": probe}
        # the last column is not feature_selection
        y = df.iloc[:, -1:]
        assert isinstance(y, pd.DataFrame)
        for category in self.attack_categories:
            # category: ddos, u2r, r2l, probe
            for c in self.attack_categories[category]:
                # c: back, buffer_overflow etc.
                y.replace(c + '.', category, inplace=True)
        y.replace("normal.", "normal", inplace=True)

        x = df.iloc[:, :-1]
        services = ['aol', 'netbios_ns', 'sql_net', 'name', 'red_i', 'icmp', 'link', 'shell', 'netstat', 'urh_i',
                    'urp_i',
                    'domain_u', 'domain', 'ftp_data', 'uucp_path', 'hostnames', 'ssh', 'finger', 'netbios_ssn', 'other',
                    'ecr_i', 'pop_3', 'kshell', 'ctf', 'whois', 'nnsp', 'http_8001', 'gopher', 'discard', 'klogin',
                    'time',
                    'iso_tsap', 'systat', 'tftp_u', 'ntp_u', 'nntp', 'telnet', 'ldap', 'remote_job', 'imap4', 'X11',
                    'courier', 'private', 'harvest', 'efs', 'uucp', 'bgp', 'tim_i', 'vmnet', 'pm_dump', 'http_2784',
                    'smtp',
                    'csnet_ns', 'mtp', 'http', 'eco_i', 'ftp', 'exec', 'rje', 'pop_2', 'supdup', 'sunrpc', 'IRC',
                    'login',
                    'echo', 'auth', 'netbios_dgm', 'http_443', 'daytime', 'Z39_50', 'printer']

        if self.encoding == self.ONE_HOT_ENCODING:
            x, y = self.one_hot_encoding(x, y, services, names)
        else:
            x, y = self.label_encoding(x, y, services)

        self.features = x.columns.values
        x = x.as_matrix()
        if isinstance(y, pd.DataFrame):
            y = y.as_matrix()

        return x, y

    @staticmethod
    def _one_hot_encoding(df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        help method for one hot encoding
        """
        for feature in features:
            one_hot = pd.get_dummies(df[feature], feature, '_')
            # And the next two statements 'replace' the existing feature_selection by the new binary-valued features
            # First, drop the existing column
            df.drop(feature, axis=1, inplace=True)
            # Next, concatenate the new columns. This assumes no clash of column names.
            df = pd.concat([df, one_hot], axis=1)
        return df

    def one_hot_encoding(self, x: pd.DataFrame, y: pd.DataFrame, services: list, names: list) -> (
            pd.DataFrame, pd.DataFrame):
        fake_data = []
        for service in services:
            fake_data.append([0, 'tcp', service, 'SF', 181, 5450, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 8, 8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9, 9, 1.0,
                              0.0, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0])
        x = pd.concat([x, pd.DataFrame(fake_data, columns=names[:-1])], ignore_index=True)
        categorical_features = ["protocol_type", "service", "flag"]
        x = self._one_hot_encoding(x, categorical_features)
        y = self._one_hot_encoding(y, ["result"])
        x = x[:-len(services)]
        return x, y

    def label_encoding(self, x: pd.DataFrame, y: pd.DataFrame, services: list) -> (pd.DataFrame, pd.DataFrame):
        le = LabelEncoder()
        le = le.fit(services)
        x['service'] = le.transform(x['service'])
        for feature in ["protocol_type", "flag"]:
            x[feature] = le.fit_transform(x[feature])
        y = le.fit_transform(y)
        print(le.classes_)
        return x, y

    def select_feature(self, x: np.ndarray, y: np.ndarray, clf) -> np.ndarray:
        if self.feature_selection == self.UNIVARIATE_FEATURES_SELECTION:
            x = self.univariate_features_selection(x, y)
        elif self.feature_selection == self.DIMENSION_REDUCTION:
            x = self.dimension_reduction(x)
        # elif self.feature_selection ==

        elif self.feature_selection == self.RECURSIVE_FEATURE_ELIMINATION:
            x = self.recursive_feature_elimination(x, y, clf)
        return x

    def dimension_reduction(self, x: np.ndarray):
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

        pca = PCA(10)
        pca.fit(x)
        # print(sorted(pca.explained_variance_ratio_, reverse=True))
        # self.plot_variance(x, pca.explained_variance_ratio_)
        x = pca.transform(x)
        pd.DataFrame(x).to_csv("merged_data_reduced.csv", index=False)
        return x

    def univariate_features_selection(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        selector = SelectKBest(chi2, k=10)
        selector = selector.fit(x, y)
        selected_features = self.features[selector.get_support()]
        print(selected_features)
        x = selector.transform(x)
        return x

    def tree_based_feature_selection(self, x, y):
        forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
        forest.fit(x, y)
        importances = forest.feature_importances_
        print(importances)
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")

        for f in range(10):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(10), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(10), indices)
        plt.xlim([-1, 10])
        plt.show()

    def recursive_feature_elimination(self, x: np.ndarray, y: np.ndarray, clf=None) -> np.ndarray:
        selector = RFECV(estimator=clf, step=1, cv=StratifiedKFold(y), scoring='accuracy')
        selector.fit(x, y)

        print("Optimal number of features : %d" % selector.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
        plt.show()

        selected_features = self.features[selector.get_support()]
        print(selected_features)
        x = selector.transform(x)
        return x

    @staticmethod
    def table_of_confusion(matrix: np.ndarray) -> list:
        table = []

        for i in range(0, matrix.shape[0]):
            tp = matrix[i, i]
            tn = matrix[i + 1:, i + 1:].sum() + matrix[:i, :i].sum()
            fn = matrix[i].sum() - tp
            fp = matrix[:, i].sum() - tp
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            false_positive_rate = fp / (fp + tn)
            false_negative_rate = fn / (fn + tp)
            table.append(
                {'sensitivity': sensitivity, 'specificity': specificity,
                 'accuracy': accuracy, 'false_positive_rate': false_positive_rate,
                 'false_negative_rate': false_negative_rate})

        return table

    def data_validation(self, x: np.ndarray, y: np.ndarray, clf, name: str):
        n = 10
        kf = StratifiedKFold(len(x), n_folds=n)
        a_scores = 0
        # create a empty matrix
        n_y = len(self.attack_categories) + 1
        total_matrix = np.zeros((n_y, n_y))

        for train_index, test_index in kf:
            x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
            clf = clf.fit(x_train, y_train)

            y_pre = clf.predict(x_test)
            a_scores += accuracy_score(y_test, y_pre)
            if self.encoding == self.LABEL_ENCODING:
                total_matrix = confusion_matrix(y_test, y_pre) + total_matrix

        print("accuracy_score for " + name + ": ")
        print(a_scores / n)
        for t in self.table_of_confusion(total_matrix):
            print(t)

    def decision_tree(self):

        clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=100, min_samples_leaf=100)
        self.x = self.select_feature(self.x, self.y, clf)
        # tree.export_graphviz(clf, out_file='decision_tree.dot')
        # cmd = "dot -Tpng decision_tree.dot -o decision_tree.png".split()
        # subprocess.call(cmd)
        self.data_validation(self.x, self.y, clf, self.decision_tree.__name__)

    def random_forest(self):
        clf = RandomForestClassifier(n_estimators=100, max_features=10, criterion='entropy', n_jobs=10,
                                     min_samples_split=100,
                                     min_samples_leaf=100)
        self.x = self.select_feature(self.x, self.y, clf)
        self.data_validation(self.x, self.y, clf, self.random_forest.__name__)

    def support_vector_classification(self):
        clf = SVC(kernel="linear")
        self.x = self.select_feature(self.x, self.y, clf)
        self.data_validation(self.x, self.y, clf, self.support_vector_classification.__name__)


if __name__ == "__main__":
    kdc = KDC("kddcup.data_10_percent_corrected", "corrected", KDC.LABEL_ENCODING, KDC.RECURSIVE_FEATURE_ELIMINATION)
    # kdc = KDC("kddcup.data_10_percent_corrected", "corrected", KDC.ONE_HOT_ENCODING, KDC.DIMENSION_REDUCTION)
    # kdc.decision_tree()
    # kdc.random_forest()
    kdc.support_vector_classification()
    # kdc = KDC("kddcup.data_10_percent_corrected", "corrected", KDC.ONE_HOT_ENCODING)
    # kdc.decision_tree()

    # kdc.random_forest()
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    # executor.submit(random_forest, train_data, test_data)
    # executor.submit(decision_tree, train_data, tet_data)
