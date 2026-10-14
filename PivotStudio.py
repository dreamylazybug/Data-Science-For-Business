"""Dataset-agnostic Excel-style pivot table builder for Analyti-Cult."""
import pandas as pd
import streamlit as st


def _mode(series):
    modes = series.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else None


def _count_unique(series):
    return series.nunique(dropna=True)


AGGREGATIONS = {
    'Count': 'count',
    'Count Unique': _count_unique,
    'Sum': 'sum',
    'Median': 'median',
    'Mean': 'mean',
    'Mode': _mode,
    'Min': 'min',
    'Max': 'max',
}


def build_pivot(df, rows, columns, values, aggregation, include_missing=False):
    """Build a pivot using explicit group keys and observed combinations only."""
    data = df.copy() if not rows else df
    synthetic = None
    if not rows and columns:
        synthetic = '__pivot_all_rows__'
        while synthetic in data.columns:
            synthetic += '_'
        data = data.assign(**{synthetic: 'All records'})
    effective_rows = list(rows) if rows else ([synthetic] if synthetic else [])
    keys = effective_rows + list(columns)
    if not keys:
        raise ValueError('Select at least one row or column grouping field.')
    if len(set(keys)) != len(keys):
        raise ValueError('Each grouping field can appear only once.')
    if not values:
        if aggregation != 'Count':
            raise ValueError('Choose a value field for this aggregation.')
        result = data.groupby(keys, dropna=not include_missing, observed=True).size().rename('Row count')
    else:
        if aggregation in ('Sum', 'Mean', 'Median'):
            bad = [c for c in values if not pd.api.types.is_numeric_dtype(df[c])]
            if bad:
                raise ValueError(f'{aggregation} requires numeric value fields. Change or remove: {", ".join(bad)}')
        grouped = data.groupby(keys, dropna=not include_missing, observed=True)
        result = grouped[list(values)].agg(AGGREGATIONS[aggregation])
        if len(values) == 1:
            result = result.rename(columns={values[0]: f'{values[0]} ({aggregation})'})
        else:
            result = result.rename(columns={c: f'{c} ({aggregation})' for c in values})
    if columns:
        result = result.unstack(level=list(range(len(effective_rows), len(keys))))
    if isinstance(result, pd.Series):
        result = result.to_frame()
    result = result.reset_index()
    if synthetic:
        result = result.drop(columns=synthetic, errors='ignore')
        if isinstance(result.columns, pd.MultiIndex):
            result = result.loc[:, [c for c in result.columns if c[0] != synthetic]]
    # Flatten MultiIndex column headers for display and CSV export.
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [' | '.join(str(x) for x in parts if str(x) not in ('', 'None')) for parts in result.columns.to_flat_index()]
    else:
        result.columns = [str(c) for c in result.columns]
    return result


def render(df, source_signature=None):
    st.subheader('Pivot Studio')
    st.caption('Create Excel-style summaries from any dataset. Choose multiple row groups, column groups, and value fields.')
    options = df.columns.tolist()
    rows = st.multiselect('Rows · group by', options, key='pivot_rows', help='Group fields appear as row labels in the chosen order.')
    columns = st.multiselect('Columns · split across', [c for c in options if c not in rows], key='pivot_columns')
    values = st.multiselect('Values · summarize', [c for c in options if c not in rows and c not in columns], key='pivot_values')
    aggregation = st.selectbox('Aggregate using', list(AGGREGATIONS), key='pivot_agg')
    include_missing = st.checkbox('Include missing grouping values', value=False, key='pivot_missing')
    if not rows and not columns:
        st.info('Choose one or more fields under Rows or Columns to build your pivot.')
        return
    if not values and aggregation != 'Count':
        st.info('Select at least one Values field, or choose Count to count records without a value field.')
        return
    # Explicit calculation prevents expensive pivots on every widget interaction.
    config = (source_signature, tuple(rows), tuple(columns), tuple(values), aggregation, include_missing)
    if st.button('Generate pivot', type='primary'):
        # Estimate width before unstacking high-cardinality column combinations.
        possible_columns = df[columns].drop_duplicates().shape[0] if columns else 1
        possible_rows = df[rows].drop_duplicates().shape[0] if rows else 1
        estimated_cells = possible_columns * max(1, len(values)) * max(1, possible_rows)
        if estimated_cells > 2_000_000 or possible_columns * max(1, len(values)) > 2000:
            st.error(f'Pivot may produce approximately {estimated_cells:,} cells. Narrow the grouping fields before generating.')
            return
        try:
            pivot = build_pivot(df, rows, columns, values, aggregation, include_missing)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f'Pivot could not be created: {exc}')
            return
        st.session_state['pivot_result'] = pivot
        st.session_state['pivot_signature'] = config
        st.success('Pivot generated and available in Visualize.')
    if st.session_state.get('pivot_signature') != config:
        st.info('Click Generate pivot to calculate the current selection.')
        return
    pivot = st.session_state.get('pivot_result')
    if pivot is None:
        return
    st.caption(f'{len(pivot):,} result rows · {pivot.shape[1]:,} result columns')
    st.dataframe(pivot.head(100), use_container_width=True, height=400)
    st.download_button('Download pivot as CSV', pivot.to_csv(index=False).encode('utf-8-sig'),
                       file_name='analyti_cult_pivot.csv', mime='text/csv')
    if st.button('Use in Visualize'):
        st.session_state['viz_requested_source'] = 'Pivot results'
        st.info('Open the Visualize tab to chart your pivot.')
