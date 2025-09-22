import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def explore_data(train_df, test_df, rul_df):
    print("==== Train Data ====")
    print(train_df.head())
    print(train_df.describe())
    print("\nMissing values:\n", train_df.isnull().sum())

    print("\n==== Test Data ====")
    print(test_df.head())
    print("\n==== RUL Data ====")
    print(rul_df.head())


def plot_sensor_trends(train_df, engine_id=1, sensors=['sensor_2', 'sensor_3', 'sensor_4']):
    """
    Plot selected sensors for a given engine over cycles.
    """
    subset = train_df[train_df['engine_id'] == engine_id]

    plt.figure(figsize=(12, 6))
    for s in sensors:
        plt.plot(subset['cycle'], subset[s], label=s)
    plt.xlabel("Cycle")
    plt.ylabel("Sensor Reading")
    plt.title(f"Engine {engine_id} - Sensor Trends")
    plt.legend()
    plt.show()


def correlation_heatmap(train_df):
    """
    Show correlation between sensors.
    """
    sensor_cols = [c for c in train_df.columns if "sensor" in c]
    corr = train_df[sensor_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm",annot=True, center=0)
    plt.title("Sensor Correlation Heatmap")
    plt.show()


def plot_rul_distribution(train_df):
    """
    Plot Remaining Useful Life (RUL) distribution in train set.
    """
    # Max cycle per engine = failure point
    max_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles['RUL'] = max_cycles['cycle'].max() - max_cycles['cycle']

    plt.figure(figsize=(8, 5))
    sns.histplot(max_cycles['RUL'], bins=30, kde=True)
    plt.title("RUL Distribution (Train Set)")
    plt.xlabel("RUL")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    from data_loading import load_fd001
    
    base_path = r"C:\Users\HP\Desktop\apa_proj\data"
    train_path = base_path + "\\train_FD001.txt"
    test_path = base_path + "\\test_FD001.txt"
    rul_path = base_path + "\\RUL_FD001.txt"

    train_df, test_df, rul_df = load_fd001(train_path, test_path, rul_path)

    explore_data(train_df, test_df, rul_df)
    plot_sensor_trends(train_df, engine_id=1)
    correlation_heatmap(train_df)
    plot_rul_distribution(train_df)
