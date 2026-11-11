import pandas as pd
from matplotlib import pyplot as plt

from .analysis import plot_distributions, plot_survival_rates, survival_summary
from .preprocessing import prepare_passengers


def run_analysis(data_path):
    passengers = pd.read_csv(data_path)
    print("数据预览")
    print(passengers.head().to_string(index=False))
    print("\n数据概况")
    passengers.info()
    print("\n数值列描述统计")
    print(passengers.describe().to_string())
    print("\n各列缺失值")
    print(passengers.isna().sum().to_string())

    prepared = prepare_passengers(passengers)
    analysis_data = prepared.assign(Sex=prepared["Sex"].map({0: "Female", 1: "Male"}))
    plot_distributions(prepared)
    for columns, title in (
        (["Sex", "Pclass"], "Survival rate by sex and class"),
        (["Age_Level", "Sex"], "Survival rate by age group and sex"),
        (["Age_Level", "Pclass"], "Survival rate by age group and class"),
    ):
        summary = survival_summary(analysis_data, columns)
        print(f"\n{title}")
        print(summary.to_string(float_format="{:.2%}".format))
        plot_survival_rates(summary, title)

    print("\n以上为描述性统计。年龄缺失值使用平均年龄填补，可能影响年龄分组结果。")
    plt.show()
