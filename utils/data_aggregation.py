import dask.dataframe as dd

def group_by_aggregate(df, group_by_cols, agg_dict):
    """Performs a groupby operation on a Dask DataFrame."""
    try:
        aggregated_df = df.groupby(group_by_cols).agg(agg_dict)
        # For Dask, we might want to compute the result here if it's expected to be small
        # For now, we return the dask df
        return aggregated_df.reset_index(), None
    except Exception as e:
        return None, f"Error during aggregation: {e}"

def create_pivot_table(df, index, columns, values, aggfunc):
    """Creates a pivot table from a Dask DataFrame."""
    # Dask's pivot_table is slightly different and might require computation
    try:
        # This is a pandas-like operation that dask supports
        pivot_df = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
        return pivot_df.reset_index(), None
    except Exception as e:
        return None, f"Error creating pivot table: {e}"
