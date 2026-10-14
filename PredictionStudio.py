"""Professional, dataset-agnostic model results UI."""
import pandas as pd
import plotly.express as px
import streamlit as st
from Custplots import polish


def render(result):
    st.subheader('Prediction Studio')
    st.caption(f"{result['algorithm']} · {result['task']} · {result['train_size']:,} training rows · {result['test_size']:,} holdout rows")
    metrics = result['metrics']
    columns = st.columns(len(metrics))
    for col, (name, value) in zip(columns, metrics.items()):
        display = f'{value:.1%}' if result['task'] == 'Classification' else f'{value:,.3f}'
        col.metric(name, display)
    predictions = result['predictions']
    if result['task'] == 'Classification':
        left, right = st.columns([1, 1])
        with left:
            counts = predictions['Predicted'].value_counts().rename_axis('Predicted class').reset_index(name='Records')
            fig = px.bar(counts, x='Predicted class', y='Records', color='Predicted class',
                         color_discrete_sequence=['#2563eb', '#14b8a6', '#f59e0b', '#8b5cf6'])
            st.plotly_chart(polish(fig, 310), use_container_width=True)
        with right:
            if result['matrix'] is not None and len(result['labels']) <= 25:
                fig = px.imshow(result['matrix'], x=result['labels'], y=result['labels'], text_auto=True,
                                color_continuous_scale='Blues', labels={'x':'Predicted', 'y':'Actual', 'color':'Records'})
                st.plotly_chart(polish(fig, 310), use_container_width=True)
            else:
                st.info('Confusion matrix hidden for high-cardinality classes.')
    else:
        left, right = st.columns(2)
        with left:
            fig = px.scatter(predictions, x='Actual', y='Predicted', opacity=.65,
                             color_discrete_sequence=['#2563eb'])
            st.plotly_chart(polish(fig, 310), use_container_width=True)
        with right:
            fig = px.histogram(predictions, x='Residual', nbins=30, color_discrete_sequence=['#14b8a6'])
            st.plotly_chart(polish(fig, 310), use_container_width=True)
    st.markdown('#### Prediction results')
    st.caption('Row IDs are generated for this export; source identifiers and feature columns are excluded. Row IDs are not reversible identifiers.')
    display = predictions.copy()
    if result['task'] == 'Classification':
        classes = sorted(display['Predicted'].astype(str).unique().tolist())
        chosen = st.multiselect('Filter predicted classes', classes, default=classes, key='pred_classes')
        display = display[display['Predicted'].astype(str).isin(chosen)]
        if 'Prediction confidence' in display:
            minimum = st.slider('Minimum prediction confidence', 0.0, 1.0, 0.0, 0.05, key='pred_confidence')
            display = display[display['Prediction confidence'] >= minimum]
        st.caption('Confidence is a model probability estimate, not a guarantee of correctness.')
    else:
        display = display.sort_values('Residual', key=lambda s: s.abs(), ascending=False)
    st.dataframe(display, use_container_width=True, hide_index=True, height=330)
    st.download_button('Download filtered predictions (CSV)', data=display.to_csv(index=False).encode('utf-8'),
                       file_name='analyti_cult_predictions.csv', mime='text/csv', key='download_predictions')
    st.caption('Evaluation metrics are measured on held-out test rows. Review data leakage, class balance and real-world suitability before using predictions in decisions.')
