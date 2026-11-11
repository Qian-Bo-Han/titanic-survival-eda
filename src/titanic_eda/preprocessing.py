import pandas as pd


def prepare_passengers(passengers):
    """填补缺失值，编码性别并新增年龄分组。"""
    prepared = passengers.copy()
    prepared["Embarked"] = prepared["Embarked"].fillna("S")
    prepared["Age"] = prepared["Age"].fillna(prepared["Age"].mean())
    prepared["Sex"] = prepared["Sex"].map({"female": 0, "male": 1})
    prepared["Age_Level"] = pd.cut(
        prepared["Age"],
        bins=[float("-inf"), 18, 40, float("inf")],
        labels=["<=18", "(18, 40]", ">40"],
    )
    return prepared
