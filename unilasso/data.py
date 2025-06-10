import pathlib
import pandas as pd

def read_data(folder: pathlib.Path):
    """Return dictionary of each cohort as a dataframe.

    Args:
        folder (pathlib.Path): Path to the folder containing the data.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing the dataframes for each cohort.
    """
    df = pd.read_csv(folder / "All_datasets_SomaScan_Plasma_7k_CVQC_SADRC_KADRC_ROSMAP_Kaci.csv")
    return {
        cohort: table for cohort, table in df.groupby("Cohort")
    }

def format_for_regression(df: pd.DataFrame, target_variable: str = "Age"):
    """Returns in X, y format for regression

    Args:
        df (pd.DataFrame): DataFrame with the data
        target_variable (str, optional): Target variable to use. Defaults to "Age".

    Returns:
        (pd.DataFrame, pd.DataFrame): X, y dataframes.
    """
    return df.iloc[:, 31:], df[target_variable]