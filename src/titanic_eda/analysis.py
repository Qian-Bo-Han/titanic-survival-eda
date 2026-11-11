from matplotlib import pyplot as plt


def survival_summary(passengers, columns):
    """统计每组乘客人数、生还人数和生还率。"""
    return passengers.groupby(columns, observed=True)["Survived"].agg(
        passengers="size", survivors="sum", survival_rate="mean"
    )


def plot_distributions(passengers):
    sex_labels = {0: "Female", 1: "Male"}
    class_labels = {1: "First", 2: "Second", 3: "Third"}

    sex_counts = passengers.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
    sex_counts = sex_counts.rename(index=sex_labels, columns={0: "Died", 1: "Survived"})
    for outcome in ("Survived", "Died"):
        _, ax = plt.subplots()
        sex_counts[outcome].plot.bar(ax=ax, title=f"{outcome} by sex", ylabel="Passengers")

    sex_counts.plot.bar(stacked=True, title="Survival counts by sex", ylabel="Passengers")

    for column, title in (
        ("Pclass", "Survival counts by passenger class"),
        ("Age_Level", "Survival counts by age group"),
    ):
        counts = (
            passengers.groupby([column, "Survived"], observed=True)
            .size()
            .unstack(fill_value=0)
            .rename(columns={0: "Died", 1: "Survived"})
        )
        if column == "Pclass":
            counts = counts.rename(index=class_labels)
        counts.plot.bar(stacked=True, title=title, ylabel="Passengers")

    _, ax = plt.subplots()
    survived = passengers["Survived"].eq(1)
    ax.hist(
        [passengers.loc[survived, "Age"], passengers.loc[~survived, "Age"]],
        stacked=True,
        label=["Survived", "Died"],
    )
    ax.set(title="Age distribution by survival", xlabel="Age", ylabel="Passengers")
    ax.legend()


def plot_survival_rates(summary, title):
    _, ax = plt.subplots()
    summary["survival_rate"].mul(100).plot.bar(
        ax=ax, title=title, ylabel="Survival rate (%)", ylim=(0, 110)
    )
    ax.bar_label(ax.containers[0], fmt="%.2f%%", padding=3)
