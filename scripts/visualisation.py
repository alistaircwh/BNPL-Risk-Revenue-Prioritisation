from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import IntegerType, LongType, DoubleType, StringType
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_top_merchants_pie(top_n_merchants, level):
    """
    Plots a pie chart of the top 15 merchants by total revenue for a given revenue level.
    top_n_merchants: pandas DataFrame with columns 'name' and 'total_revenue', pre-filtered to top 15.
    level: revenue level label (e.g. 'a') used in the chart title.
    Displays the chart inline; returns None.
    """
    colors = [ "#FF6384",  "#36A2EB",  "#FFCE56",  "#4BC0C0",  "#9966FF",  "#FF9F40",  # Orange
    "#C9CBCF",  "#2ECC71",  "#E74C3C",  "#3498DB",  "#9B59B6",  "#F39C12",  "#1ABC9C",  "#34495E",  "#95A5A6"   # Silver
    ]

    if not top_n_merchants.empty:
        plt.figure(figsize=(10, 8))
        plt.pie(top_n_merchants["total_revenue"], labels=top_n_merchants["name"], colors = colors,
                autopct='%1.1f%%', startangle=140, explode= [0.05] * 15)
        plt.title(f"Top 15 Companies by Total Revenue in Revenue Level '{level}'", pad = 20)
        plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        plt.show()
