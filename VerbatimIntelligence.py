"""Dataset-agnostic, evidence-linked text analysis. No external model required."""
import re
from collections import Counter
import pandas as pd
import streamlit as st

STOP = set('the a an and or but for from with this that these those is are was were be been being to of in on at by as it its we our you your they their i me my not do does did have has had can could would should very just about into over under after before when where what why how than then so if there here all any more most much many also one two per via'.split())


def text_candidates(df):
    return [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]


def analyze_verbatim(df, columns, group_col=None, min_word_length=3, top_n=20, max_examples=3):
    if not columns:
        raise ValueError('Choose at least one text column.')
    if any(c not in df for c in columns):
        raise ValueError('Text column not found in dataset.')
    records = []
    for column in columns:
        for idx, val in df[column].dropna().items():
            if not isinstance(val, str) or not val.strip():
                continue
            text = val.strip()
            if len(text) > 20000:
                text = text[:20000] + ' [truncated]'
            records.append({'source_row': str(idx), 'column': column, 'comment': text,
                            'segment': str(df.at[idx, group_col]) if group_col else 'All records'})
    comments = pd.DataFrame(records, columns=['source_row','column','comment','segment'])
    if comments.empty:
        return {'comments': comments, 'keywords': pd.DataFrame(columns=['term','mentions','percent_of_comments']), 'segments': pd.DataFrame(), 'total': 0, 'method': 'Transparent keyword frequencies (not semantic topic modeling)'}
    token_sets = []
    for text in comments['comment']:
        tokens = set(w.lower() for w in re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text) if len(w) >= min_word_length and w.lower() not in STOP)
        token_sets.append(tokens)
    counts = Counter(w for tokens in token_sets for w in tokens)
    keywords = pd.DataFrame([{'term': word, 'mentions': n, 'percent_of_comments': round(100*n/len(comments), 1)}
                             for word, n in counts.most_common(top_n)])
    if group_col:
        segments = comments.groupby('segment', dropna=False).size().rename('comments').reset_index().sort_values('comments', ascending=False)
    else:
        segments = pd.DataFrame()
    return {'comments': comments, 'keywords': keywords, 'segments': segments,
            'total': len(comments), 'method': 'Transparent keyword frequencies (not semantic topic modeling)'}


def render(df, source_key=None):
    st.subheader('Verbatim Intelligence')
    st.caption('Explore recurring words and representative source comments. This offline MVP does not claim to infer sentiment or semantic themes.')
    fields = text_candidates(df)
    if not fields:
        st.info('No text columns found. Upload a dataset containing notes, comments, or feedback.')
        return
    suggested = [c for c in fields if any(k in c.lower() for k in ['note','comment','feedback','reason','description','verbatim','review','text'])]
    with st.form('verbatim_config'):
        columns = st.multiselect('Text columns to analyze', fields, default=suggested[:2], help='Search and select one or more text fields.')
        group = st.selectbox('Compare comment volume by (optional)', ['None'] + [c for c in df.columns if c not in columns])
        top_n = st.slider('Top terms', 5, 50, 20)
        run = st.form_submit_button('Analyze comments', type='primary')
    config = (source_key, tuple(columns), group, top_n)
    if run:
        if group != 'None' and df[group].nunique(dropna=False) > 100:
            st.warning('Grouping column has more than 100 distinct values; choose a lower-cardinality segment.')
        try:
            st.session_state['verbatim_result'] = analyze_verbatim(df, columns, None if group == 'None' else group, top_n=top_n)
            st.session_state['verbatim_config'] = config
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    if st.session_state.get('verbatim_config') != config:
        st.info('Choose your columns and click Analyze comments.')
        return
    result = st.session_state.get('verbatim_result')
    if result is None:
        return
    st.metric('Non-empty text entries', f"{result['total']:,}")
    st.caption('A row can contribute more than one entry when multiple text columns are selected. Counts represent comments, not unique customers.')
    st.markdown('#### Most frequently mentioned terms')
    st.dataframe(result['keywords'], use_container_width=True, hide_index=True)
    if not result['keywords'].empty:
        st.bar_chart(result['keywords'].set_index('term')['mentions'])
    if not result['segments'].empty:
        st.markdown('#### Comment volume by segment')
        st.dataframe(result['segments'].head(100), use_container_width=True, hide_index=True)
    st.markdown('#### Source comments')
    term = st.text_input('Find comments containing a term', key='verbatim_term')
    shown = result['comments']
    if term:
        shown = shown[shown['comment'].str.contains(term, case=False, regex=False, na=False)]
    st.dataframe(shown.head(100), use_container_width=True, hide_index=True)
    st.download_button('Download analyzed comments CSV', result['comments'].to_csv(index=False).encode('utf-8-sig'),
                       file_name='verbatim_comments.csv', mime='text/csv')
    st.session_state['verbatim_latest_result'] = result
