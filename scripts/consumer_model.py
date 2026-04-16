import matplotlib.pyplot as plt
import seaborn as sns



def feature_visualisation(df_pandas, plots):
    """
    Renders a grid of feature plots for consumer-level data exploration.
    df_pandas: pandas DataFrame containing the features to visualise.
    plots: dict mapping plot title → (feature_column, plot_type), where plot_type is
           one of 'hist', 'count', 'scatter1'–'scatter4' (income proportion scatter plots).
    Displays the plot inline; returns None.
    """
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))  
    fig.tight_layout(pad=5.0) 
    for i, (plot_title, (feature, plot_type)) in enumerate(plots.items()):
        ax = axes[i // 3, i % 3] 
        if plot_type == "hist":
            sns.histplot(df_pandas[feature], bins=30, kde=True, ax=ax)
        elif plot_type == "count":
            sns.countplot(x=feature, data=df_pandas, ax=ax)
        elif plot_type.startswith("scatter"):
            if plot_type == "scatter1":
                sns.scatterplot(x="Proportion_between_max_order_value_mean_income", y="average_fraud_probability", data=df_pandas, ax=ax)
            elif plot_type == "scatter2":
                sns.scatterplot(x="Proportion_between_max_order_value_median_income", y="average_fraud_probability", data=df_pandas, ax=ax)
            elif plot_type == "scatter3":
                sns.scatterplot(x="Proportion_between_total_order_value_mean_income", y="average_fraud_probability", data=df_pandas, ax=ax)
            elif plot_type == "scatter4":
                sns.scatterplot(x="Proportion_between_total_order_value_median_income", y="average_fraud_probability", data=df_pandas, ax=ax)

        ax.set_title(plot_title)

    plt.show()

