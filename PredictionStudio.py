"""Dataset-agnostic evaluation dashboard with binary threshold and class diagnostics."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             classification_report, precision_recall_fscore_support)
from Custplots import polish


def render(result):
    st.subheader('Prediction Studio')
    st.caption(f"{result['algorithm']} · {result['task']} · {result['train_size']:,} training rows · {result['test_size']:,} holdout rows")
    predictions = result['predictions'].copy()
    metrics = result['metrics'].copy()
    labels = result['labels']
    matrix = result['matrix']
    report = result.get('class_report')
    if result['task'] == 'Classification':
        st.markdown('#### Classification diagnostics')
        actual_counts = predictions['Actual'].value_counts().reindex(labels, fill_value=0)
        st.caption('Actual holdout class counts: ' + ' · '.join(f'{name}: {int(count):,}' for name, count in actual_counts.items()))
        if actual_counts.max() >= 2 * max(1, actual_counts.min()):
            st.warning('The holdout classes are imbalanced. Evaluate per-class recall and precision, not just accuracy.')
        probs = result.get('binary_probabilities')
        if probs is not None:
            positive = labels[1]
            threshold = st.slider(f'Decision threshold for class “{positive}”',
                                  min_value=0.05, max_value=0.95, value=0.50, step=0.05,
                                  help='Adjusts holdout predictions without retraining. Lower thresholds often increase recall and false positives.',
                                  key='prediction_threshold')
            pred_ids = (np.asarray(probs) >= threshold).astype(int)
            actual_ids = np.asarray(result['binary_actual'])
            predictions['Predicted'] = np.asarray(labels)[pred_ids]
            predictions['Prediction confidence'] = np.where(pred_ids == 1, probs, 1 - np.asarray(probs))
            matrix = confusion_matrix(actual_ids, pred_ids, labels=[0, 1])
            precision, recall, f1, _ = precision_recall_fscore_support(actual_ids, pred_ids, average='weighted', zero_division=0)
            metrics = {'Accuracy': accuracy_score(actual_ids, pred_ids),
                       'Balanced accuracy': balanced_accuracy_score(actual_ids, pred_ids),
                       'Weighted precision': precision, 'Weighted recall': recall, 'Weighted F1': f1}
            report = pd.DataFrame(classification_report(actual_ids, pred_ids, labels=[0, 1],
                         target_names=labels, output_dict=True, zero_division=0)).T.loc[labels]
            st.caption('Threshold is applied to held-out test predictions for exploration. Choose a production threshold using separate validation data.')
        else:
            metrics = {'Accuracy': metrics['Accuracy'], 'Balanced accuracy': metrics['Balanced accuracy'],
                       'Weighted precision': metrics['Precision'], 'Weighted recall': metrics['Recall'],
                       'Weighted F1': metrics['F1']}
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f'{value:.1%}')
        if report is not None:
            st.markdown('#### Performance by class')
            show = report.rename(columns={'precision':'Precision', 'recall':'Recall', 'f1-score':'F1', 'support':'Support'}).copy()
            show.index.name = 'Class'
            st.dataframe(show.reset_index().style.format({'Precision':'{:.1%}', 'Recall':'{:.1%}', 'F1':'{:.1%}', 'Support':'{:.0f}'}),
                         use_container_width=True, hide_index=True)
            missed = [str(name) for name in show.index if show.loc[name, 'Recall'] == 0 and show.loc[name, 'Support'] > 0]
            if missed:
                st.error('No actual cases detected for: ' + ', '.join(missed))
        left, right = st.columns(2)
        with left:
            counts = predictions['Predicted'].value_counts().reindex(labels, fill_value=0).rename_axis('Predicted class').reset_index(name='Records')
            fig = px.bar(counts, x='Predicted class', y='Records', color='Predicted class',
                         color_discrete_sequence=['#2563eb','#14b8a6','#f59e0b','#8b5cf6'])
            st.plotly_chart(polish(fig, 310), use_container_width=True)
        with right:
            if matrix is not None and len(labels) <= 25:
                fig = px.imshow(matrix, x=labels, y=labels, text_auto=True, color_continuous_scale='Blues',
                                labels={'x':'Predicted','y':'Actual','color':'Records'})
                st.plotly_chart(polish(fig, 310), use_container_width=True)
    else:
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f'{value:,.3f}')
        left, right = st.columns(2)
        with left:
            st.plotly_chart(polish(px.scatter(predictions, x='Actual', y='Predicted', opacity=.65),310), use_container_width=True)
        with right:
            st.plotly_chart(polish(px.histogram(predictions, x='Residual', nbins=30),310), use_container_width=True)
    st.markdown('#### Prediction results')
    st.caption('Generated row IDs are not reversible identifiers. Source identifiers and predictors are excluded from exports.')
    display = predictions.copy()
    if result['task'] == 'Classification':
        classes = labels
        chosen = st.multiselect('Filter predicted classes', classes, default=classes, key='pred_classes')
        display = display[display['Predicted'].astype(str).isin(chosen)]
        if 'Prediction confidence' in display:
            minimum = st.slider('Minimum prediction confidence', 0.0, 1.0, 0.0, 0.05, key='pred_confidence')
            display = display[display['Prediction confidence'] >= minimum]
        st.caption('Probabilities are model estimates, not guarantees. Threshold changes affect the displayed predictions and metrics.')
    else:
        display = display.sort_values('Residual', key=lambda s: s.abs(), ascending=False)
    st.dataframe(display, use_container_width=True, hide_index=True, height=330)
    st.download_button('Download filtered predictions (CSV)', data=display.to_csv(index=False).encode('utf-8'),
                       file_name='analyti_cult_predictions.csv', mime='text/csv', key='download_predictions')
    st.caption('Holdout metrics are for evaluation. Check data leakage, calibration, subgroup performance, and deployment suitability.')
