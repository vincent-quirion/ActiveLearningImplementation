import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Experience:
    def __init__(self):
        self.tests_list = []

    def add_test(self, acquisition_fn, nb_examples, accuracy):
        self.tests_list.append(
            {"Acquisition Function": acquisition_fn, "Labeled Examples": nb_examples, "Accuracy": accuracy}
        )

    def save_test_plot(self, path):
        df = pd.DataFrame.from_records(self.tests_list)
        sns.relplot(data=df, x="Labeled Examples", y="Accuracy", hue="Acquisition Function", kind="line")
        plt.savefig(path)
