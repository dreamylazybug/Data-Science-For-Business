"""Dataset-agnostic visualization workspace with composable filters."""
import re
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from Custplots import polish, cstplt


def field_kind(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return 'Date'
    if pd.api.types.is_bool_dtype(s):
        return 'Boolean'
    if pd.api.types.is_numeric_dtype(s):
        return 'Number'
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(150)
        if len(sample) and (sample.str.match(r'^\d{4}-\d{2}-\d{2}').mean() > .9):
            return 'Date'
        if s.nunique(dropna=True) > 40 and s.nunique(dropna=True) / max(len(s), 1) > .5:
            return 'Text'
    return 'Category'


def filter_frame(df, rules, join='AND'):
    """Apply filters without mutating source data; supports AND/OR."""
    if not rules:
        return df.copy()
    masks = []
    for rule in rules:
        name, op, value = rule['field'], rule['op'], rule.get('value')
        if name not in df:
            continue
        s = df[name]
        if op == 'is missing':
            mask = s.isna()
        elif op == 'is not missing':
            mask = s.notna()
        elif op == 'in':
            mask = s.astype('string').isin([str(v) for v in (value or [])])
        elif op == 'between':
            low, high = value
            mask = pd.to_numeric(s, errors='coerce').between(low, high)
        elif op == 'date between':
            low, high = value
            mask = pd.to_datetime(s, errors='coerce').between(pd.Timestamp(low), pd.Timestamp(high))
        elif op == 'contains':
            mask = s.astype('string').str.contains(str(value), case=False, regex=False, na=False)
        elif op == 'starts with':
            mask = s.astype('string').str.startswith(str(value), na=False)
        else:
            continue
        masks.append(mask.fillna(False).to_numpy(dtype=bool))
    if not masks:
        return df.copy()
    combined = np.logical_and.reduce(masks) if join == 'AND' else np.logical_or.reduce(masks)
    return df.loc[combined].copy()


def _filter_editor(df, prefix='viz'):
    cols = list(df.columns)
    if not cols:
        return []
    st.markdown('#### 2 · Filter data')
    with st.expander('Filter builder · add as many columns as needed', expanded=True):
        chosen = st.multiselect('Filter columns (type to search)', cols, key=f'{prefix}_filter_fields',
                                help='Search by column name. Each selected column gets a type-aware filter.')
        join = st.radio('Combine filters', ['AND', 'OR'], horizontal=True, key=f'{prefix}_join')
        rules = []
        for i, field in enumerate(chosen):
            s = df[field]
            kind = field_kind(s)
            with st.container(border=True):
                st.caption(f'{field} · {kind}')
                if kind == 'Number':
                    op = st.selectbox('Condition', ['between', 'is missing', 'is not missing'], key=f'{prefix}_{i}_numop')
                    if op == 'between':
                        numbers = pd.to_numeric(s, errors='coerce').dropna()
                        if numbers.empty:
                            continue
                        lo, hi = float(numbers.min()), float(numbers.max())
                        a, b = st.columns(2)
                        low = a.number_input('Minimum', value=lo, key=f'{prefix}_{i}_low')
                        high = b.number_input('Maximum', value=hi, key=f'{prefix}_{i}_high')
                        if low > high:
                            st.warning('Minimum cannot exceed maximum.')
                            continue
                        value = (low, high)
                    else:
                        value = None
                elif kind == 'Date':
                    op = st.selectbox('Condition', ['date between', 'is missing', 'is not missing'], key=f'{prefix}_{i}_dateop')
                    value = None
                    if op == 'date between':
                        dates = pd.to_datetime(s, errors='coerce').dropna()
                        if dates.empty:
                            continue
                        selected = st.date_input('Date range', value=(dates.min().date(), dates.max().date()), key=f'{prefix}_{i}_dates')
                        if not isinstance(selected, (list, tuple)) or len(selected) != 2:
                            st.info('Choose a start and end date.')
                            continue
                        value = selected
                elif kind in ('Category', 'Boolean'):
                    op = st.selectbox('Condition', ['in', 'is missing', 'is not missing'], key=f'{prefix}_{i}_catop')
                    value = None
                    if op == 'in':
                        cardinality = s.nunique(dropna=True)
                        if cardinality > 2000:
                            st.warning(f'{cardinality:,} distinct values; use a text filter instead for this field.')
                            query = st.text_input('Contains text', key=f'{prefix}_{i}_query')
                            if not query:
                                continue
                            op, value = 'contains', query
                        else:
                            opts = sorted(s.dropna().astype(str).unique().tolist())
                            value = st.multiselect('Include values (searchable)', opts, key=f'{prefix}_{i}_categories')
                            if not value:
                                continue
                else:
                    op = st.selectbox('Condition', ['contains', 'starts with', 'is missing', 'is not missing'], key=f'{prefix}_{i}_textop')
                    value = None
                    if op in ('contains', 'starts with'):
                        value = st.text_input('Text', key=f'{prefix}_{i}_text')
                        if not value:
                            continue
                rules.append({'field': field, 'op': op, 'value': value})
        if st.button('Clear all filters', key=f'{prefix}_clear'):
            st.session_state[f'{prefix}_filter_fields'] = []
            st.rerun()
    return rules, join


def _plot(df, source):
    st.markdown('#### 3 · Build chart')
    if df.empty:
        st.warning('No records match your filters.')
        return
    numeric = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()
    chart = st.selectbox('Chart type', ['Bar', 'Line', 'Pie', 'Histogram', 'Scatter', 'Box', 'Correlation heatmap',
                                      'Distribution (classic)', 'Linear Regression (classic)'], key='viz_chart')
    if chart in ('Distribution (classic)', 'Linear Regression (classic)'):
        cstplt(df, 'Distribution' if chart.startswith('Distribution') else 'Linear Regression')
        return
    if chart == 'Correlation heatmap':
        if len(numeric) < 2:
            st.info('At least two numeric columns are required.')
            return
        selected = st.multiselect('Numeric columns', numeric, default=numeric[:min(12,len(numeric))], key='viz_corr')
        if len(selected) < 2:
            return
        fig = px.imshow(df[selected].corr(numeric_only=True), text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    elif chart == 'Histogram':
        x = st.selectbox('Field', all_cols, key='viz_hist_x')
        fig = px.histogram(df, x=x, nbins=35)
    elif chart in ('Scatter', 'Box'):
        if not numeric:
            st.info('Select a dataset with numeric columns.')
            return
        a,b = st.columns(2)
        x = a.selectbox('X / group', all_cols, key='viz_xy_x')
        y = b.selectbox('Y (numeric)', numeric, key='viz_xy_y')
        plot_df = df.sample(n=min(5000,len(df)), random_state=42) if chart == 'Scatter' else df
        fig = px.scatter(plot_df, x=x, y=y, opacity=.65) if chart == 'Scatter' else px.box(plot_df, x=x, y=y)
    else:
        a,b = st.columns(2)
        x = a.selectbox('X / category', all_cols, key='viz_group_x')
        if chart == 'Pie':
            y_options = ['Record count'] + numeric
            y = b.selectbox('Measure', y_options, key='viz_pie_y')
            if y == 'Record count':
                plot_df = df.groupby(x, dropna=False, observed=True).size().reset_index(name='Record count')
            else:
                plot_df = df.groupby(x, dropna=False, observed=True)[y].sum().reset_index()
            if len(plot_df) > 25:
                st.warning('Pie charts work best with 25 or fewer categories; showing top 25.')
                plot_df = plot_df.nlargest(25, y)
            fig = px.pie(plot_df, names=x, values=y)
        else:
            y = b.selectbox('Y / measure', ['Record count'] + numeric, key='viz_group_y')
            if y == 'Record count':
                plot_df = df.groupby(x, dropna=False, observed=True).size().reset_index(name='Record count')
            else:
                aggregation = st.selectbox('Aggregate measure', ['Sum','Mean','Median','Min','Max'], key='viz_group_agg')
                plot_df = df.groupby(x, dropna=False, observed=True)[y].agg(aggregation.lower()).reset_index()
            if len(plot_df) > 100:
                st.warning('Showing the first 100 groups to keep the chart readable.')
                plot_df = plot_df.head(100)
            fig = px.bar(plot_df, x=x, y=y) if chart == 'Bar' else px.line(plot_df, x=x, y=y, markers=True)
    st.plotly_chart(polish(fig, 420), use_container_width=True)
    st.download_button('Download chart data (CSV)', df.to_csv(index=False).encode('utf-8-sig'),
                       'visualize_filtered_data.csv', 'text/csv', key='viz_download')


def render(original, cleaned=None, pivot=None, pivot_metadata=None):
    st.subheader('Visualize')
    st.caption('Choose a source, add any number of filters, then build a chart. Filters affect only this workspace.')
    sources = {'Original dataset': original}
    if cleaned is not None:
        sources['Cleaned dataset'] = cleaned
    if pivot is not None:
        sources['Pivot results'] = pivot
    if st.session_state.get('viz_requested_source') == 'Pivot results' and 'Pivot results' in sources:
        st.session_state['viz_source'] = 'Pivot results'
        st.session_state.pop('viz_requested_source', None)
    st.markdown('#### 1 · Choose data source')
    selected = st.selectbox('Data source', list(sources), key='viz_source')
    data = sources[selected]
    if selected == 'Pivot results':
        st.info('Pivot filters operate on aggregated results. Filter source records before aggregation in Pivot Studio.')
    rules, join = _filter_editor(data)
    filtered = filter_frame(data, rules, join)
    a,b,c = st.columns(3)
    a.metric('Source rows', f'{len(data):,}')
    b.metric('After filters', f'{len(filtered):,}')
    c.metric('Columns', f'{filtered.shape[1]:,}')
    with st.expander('Preview filtered records'):
        st.dataframe(filtered.head(100), use_container_width=True)
    _plot(filtered, selected)
