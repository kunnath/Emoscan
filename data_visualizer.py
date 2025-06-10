import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

class DataVisualizer:
    def __init__(self):
        self.emotion_colors = {
            'happy': '#FFD700',
            'sad': '#4169E1',
            'angry': '#FF4500',
            'fear': '#8B008B',
            'surprise': '#FF69B4',
            'disgust': '#32CD32',
            'neutral': '#808080'
        }
        
        self.posture_colors = {
            'upright': '#00FF00',
            'slouched': '#FF4500',
            'leaning_left': '#FFA500',
            'leaning_right': '#FFA500',
            'head_down': '#FF6B6B',
            'arms_crossed': '#9370DB',
            'hands_on_hips': '#20B2AA',
            'relaxed': '#98FB98'
        }
    
    def create_emotion_statistics(self, data_df):
        """
        Create comprehensive emotion statistics and visualizations
        """
        if data_df.empty:
            st.info("No emotion data available")
            return
        
        try:
            # Extract emotion data
            emotion_data = []
            for _, row in data_df.iterrows():
                if 'emotions' in row and row['emotions']:
                    emotions = row['emotions']
                    if 'dominant_emotion' in emotions:
                        emotion_data.append({
                            'timestamp': row['timestamp'],
                            'emotion': emotions['dominant_emotion'],
                            'confidence': emotions.get('confidence', 0),
                            'all_emotions': emotions.get('all_emotions', {})
                        })
            
            if not emotion_data:
                st.info("No valid emotion data found")
                return
            
            emotion_df = pd.DataFrame(emotion_data)
            
            # Emotion distribution pie chart
            self._create_emotion_pie_chart(emotion_df)
            
            # Emotion timeline
            self._create_emotion_timeline(emotion_df)
            
            # Emotion confidence histogram
            self._create_emotion_confidence_chart(emotion_df)
            
            # Emotion heatmap
            self._create_emotion_intensity_heatmap(emotion_df)
            
            # Summary statistics
            self._display_emotion_summary_stats(emotion_df)
            
        except Exception as e:
            st.error(f"Error creating emotion statistics: {str(e)}")
            logging.error(f"Error in emotion statistics: {str(e)}")
    
    def create_posture_statistics(self, data_df):
        """
        Create comprehensive posture statistics and visualizations
        """
        if data_df.empty:
            st.info("No posture data available")
            return
        
        try:
            # Extract posture data
            posture_data = []
            for _, row in data_df.iterrows():
                if 'posture' in row and row['posture']:
                    posture = row['posture']
                    if 'posture_type' in posture:
                        pose_analysis = posture.get('pose_analysis', {})
                        posture_data.append({
                            'timestamp': row['timestamp'],
                            'posture_type': posture['posture_type'],
                            'confidence': posture.get('confidence', 0),
                            'posture_score': pose_analysis.get('posture_score', 0),
                            'shoulder_tilt': pose_analysis.get('shoulder_tilt', 0),
                            'head_offset_x': pose_analysis.get('head_offset_x', 0),
                            'head_forward': pose_analysis.get('head_forward', False)
                        })
            
            if not posture_data:
                st.info("No valid posture data found")
                return
            
            posture_df = pd.DataFrame(posture_data)
            
            # Posture distribution
            self._create_posture_distribution_chart(posture_df)
            
            # Posture score timeline
            self._create_posture_score_timeline(posture_df)
            
            # Posture quality gauge
            self._create_posture_quality_gauge(posture_df)
            
            # Posture analysis details
            self._create_posture_analysis_charts(posture_df)
            
            # Summary statistics
            self._display_posture_summary_stats(posture_df)
            
        except Exception as e:
            st.error(f"Error creating posture statistics: {str(e)}")
            logging.error(f"Error in posture statistics: {str(e)}")
    
    def _create_emotion_pie_chart(self, emotion_df):
        """Create emotion distribution pie chart"""
        emotion_counts = emotion_df['emotion'].value_counts()
        
        fig = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="Emotion Distribution",
            color=emotion_counts.index,
            color_discrete_map=self.emotion_colors
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_emotion_timeline(self, emotion_df):
        """Create emotion timeline chart"""
        # Create numeric mapping for emotions
        emotion_mapping = {emotion: i for i, emotion in enumerate(emotion_df['emotion'].unique())}
        emotion_df['emotion_numeric'] = emotion_df['emotion'].map(emotion_mapping)
        
        fig = px.scatter(
            emotion_df,
            x='timestamp',
            y='emotion',
            color='emotion',
            size='confidence',
            title="Emotion Timeline",
            color_discrete_map=self.emotion_colors,
            hover_data=['confidence']
        )
        
        fig.update_layout(height=400)
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_emotion_confidence_chart(self, emotion_df):
        """Create emotion confidence histogram"""
        fig = px.histogram(
            emotion_df,
            x='confidence',
            color='emotion',
            title="Emotion Detection Confidence Distribution",
            color_discrete_map=self.emotion_colors,
            nbins=20
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_emotion_intensity_heatmap(self, emotion_df):
        """Create emotion intensity heatmap over time"""
        # Group by time periods and emotion
        emotion_df['hour'] = pd.to_datetime(emotion_df['timestamp']).dt.hour
        emotion_df['minute_group'] = pd.to_datetime(emotion_df['timestamp']).dt.minute // 5 * 5
        
        # Create pivot table
        pivot_data = emotion_df.groupby(['hour', 'emotion']).size().unstack(fill_value=0)
        
        if not pivot_data.empty:
            fig = px.imshow(
                pivot_data.T,
                title="Emotion Intensity Heatmap by Time",
                labels=dict(x="Hour", y="Emotion", color="Count"),
                color_continuous_scale="Viridis"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_emotion_summary_stats(self, emotion_df):
        """Display emotion summary statistics"""
        st.markdown("#### 📊 Emotion Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            most_common = emotion_df['emotion'].mode().iloc[0] if not emotion_df.empty else "N/A"
            st.metric("Most Common Emotion", most_common)
        
        with col2:
            avg_confidence = emotion_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            total_detections = len(emotion_df)
            st.metric("Total Detections", total_detections)
        
        with col4:
            unique_emotions = emotion_df['emotion'].nunique()
            st.metric("Unique Emotions", unique_emotions)
        
        # Detailed emotion breakdown
        st.markdown("#### 📈 Detailed Emotion Breakdown")
        emotion_stats = emotion_df.groupby('emotion').agg({
            'confidence': ['mean', 'std', 'count'],
            'timestamp': ['min', 'max']
        }).round(3)
        
        emotion_stats.columns = ['Avg Confidence', 'Std Confidence', 'Count', 'First Seen', 'Last Seen']
        st.dataframe(emotion_stats)
    
    def _create_posture_distribution_chart(self, posture_df):
        """Create posture type distribution chart"""
        posture_counts = posture_df['posture_type'].value_counts()
        
        fig = px.bar(
            x=posture_counts.index,
            y=posture_counts.values,
            title="Posture Type Distribution",
            color=posture_counts.index,
            color_discrete_map=self.posture_colors
        )
        
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_posture_score_timeline(self, posture_df):
        """Create posture score timeline"""
        fig = px.line(
            posture_df,
            x='timestamp',
            y='posture_score',
            title="Posture Score Over Time",
            color_discrete_sequence=['#2E86AB']
        )
        
        # Add threshold lines
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Good Posture")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Fair Posture")
        fig.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Poor Posture")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_posture_quality_gauge(self, posture_df):
        """Create posture quality gauge"""
        avg_score = posture_df['posture_score'].mean()
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Posture Score"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_posture_analysis_charts(self, posture_df):
        """Create detailed posture analysis charts"""
        # Shoulder tilt analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                posture_df,
                x='shoulder_tilt',
                title="Shoulder Tilt Distribution",
                nbins=20,
                color_discrete_sequence=['#FF6B6B']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="green", annotation_text="Aligned")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                posture_df,
                x='head_offset_x',
                y='posture_score',
                color='posture_type',
                title="Head Position vs Posture Score",
                color_discrete_map=self.posture_colors
            )
            fig.add_vline(x=0, line_dash="dash", line_color="green", annotation_text="Centered")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_posture_summary_stats(self, posture_df):
        """Display posture summary statistics"""
        st.markdown("#### 🤸 Posture Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            most_common = posture_df['posture_type'].mode().iloc[0] if not posture_df.empty else "N/A"
            st.metric("Most Common Posture", most_common)
        
        with col2:
            avg_score = posture_df['posture_score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}/100")
        
        with col3:
            avg_confidence = posture_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col4:
            good_posture_pct = (posture_df['posture_score'] > 80).mean() * 100
            st.metric("Good Posture %", f"{good_posture_pct:.1f}%")
        
        # Detailed posture breakdown
        st.markdown("#### 📊 Detailed Posture Analysis")
        posture_stats = posture_df.groupby('posture_type').agg({
            'posture_score': ['mean', 'std', 'count'],
            'confidence': ['mean'],
            'timestamp': ['min', 'max']
        }).round(3)
        
        posture_stats.columns = ['Avg Score', 'Std Score', 'Count', 'Avg Confidence', 'First Seen', 'Last Seen']
        st.dataframe(posture_stats)
    
    def create_combined_analysis_dashboard(self, data_df):
        """
        Create a combined dashboard showing emotion and posture correlations
        """
        if data_df.empty:
            st.info("No data available for combined analysis")
            return
        
        st.subheader("🔄 Combined Emotion & Posture Analysis")
        
        try:
            # Extract combined data
            combined_data = []
            for _, row in data_df.iterrows():
                if 'emotions' in row and 'posture' in row and row['emotions'] and row['posture']:
                    emotions = row['emotions']
                    posture = row['posture']
                    
                    if 'dominant_emotion' in emotions and 'posture_type' in posture:
                        combined_data.append({
                            'timestamp': row['timestamp'],
                            'emotion': emotions['dominant_emotion'],
                            'emotion_confidence': emotions.get('confidence', 0),
                            'posture_type': posture['posture_type'],
                            'posture_score': posture.get('pose_analysis', {}).get('posture_score', 0),
                            'posture_confidence': posture.get('confidence', 0)
                        })
            
            if not combined_data:
                st.info("No combined emotion and posture data available")
                return
            
            combined_df = pd.DataFrame(combined_data)
            
            # Emotion-Posture correlation heatmap
            self._create_emotion_posture_correlation(combined_df)
            
            # Timeline showing both metrics
            self._create_combined_timeline(combined_df)
            
            # Correlation analysis
            self._display_correlation_analysis(combined_df)
            
        except Exception as e:
            st.error(f"Error creating combined analysis: {str(e)}")
            logging.error(f"Error in combined analysis: {str(e)}")
    
    def _create_emotion_posture_correlation(self, combined_df):
        """Create emotion-posture correlation heatmap"""
        # Create cross-tabulation
        crosstab = pd.crosstab(combined_df['emotion'], combined_df['posture_type'])
        
        fig = px.imshow(
            crosstab.values,
            x=crosstab.columns,
            y=crosstab.index,
            title="Emotion-Posture Correlation Heatmap",
            color_continuous_scale="Blues",
            labels=dict(color="Frequency")
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_combined_timeline(self, combined_df):
        """Create combined emotion and posture timeline"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Emotion Timeline', 'Posture Score Timeline'),
            vertical_spacing=0.1
        )
        
        # Add emotion timeline
        for emotion in combined_df['emotion'].unique():
            emotion_data = combined_df[combined_df['emotion'] == emotion]
            fig.add_trace(
                go.Scatter(
                    x=emotion_data['timestamp'],
                    y=[emotion] * len(emotion_data),
                    mode='markers',
                    name=f'Emotion: {emotion}',
                    marker=dict(color=self.emotion_colors.get(emotion, '#808080')),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add posture score timeline
        fig.add_trace(
            go.Scatter(
                x=combined_df['timestamp'],
                y=combined_df['posture_score'],
                mode='lines+markers',
                name='Posture Score',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Combined Emotion & Posture Timeline")
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_correlation_analysis(self, combined_df):
        """Display correlation analysis between emotions and posture"""
        st.markdown("#### 🔍 Correlation Analysis")
        
        # Calculate correlations
        correlations = []
        
        for emotion in combined_df['emotion'].unique():
            emotion_data = combined_df[combined_df['emotion'] == emotion]
            if len(emotion_data) > 1:
                avg_posture_score = emotion_data['posture_score'].mean()
                std_posture_score = emotion_data['posture_score'].std()
                count = len(emotion_data)
                
                correlations.append({
                    'Emotion': emotion,
                    'Avg Posture Score': round(avg_posture_score, 1),
                    'Std Dev': round(std_posture_score, 1),
                    'Count': count
                })
        
        if correlations:
            correlation_df = pd.DataFrame(correlations)
            st.dataframe(correlation_df)
            
            # Insights
            best_posture_emotion = correlation_df.loc[correlation_df['Avg Posture Score'].idxmax(), 'Emotion']
            worst_posture_emotion = correlation_df.loc[correlation_df['Avg Posture Score'].idxmin(), 'Emotion']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Posture with Emotion", best_posture_emotion)
            with col2:
                st.metric("Worst Posture with Emotion", worst_posture_emotion)
    
    def export_data_to_csv(self, data_df, filename="emoscan_data.csv"):
        """
        Export analysis data to CSV format
        """
        try:
            # Flatten the nested data structure
            flattened_data = []
            
            for _, row in data_df.iterrows():
                flat_row = {'timestamp': row['timestamp']}
                
                # Flatten emotions data
                if 'emotions' in row and row['emotions']:
                    emotions = row['emotions']
                    flat_row.update({
                        'dominant_emotion': emotions.get('dominant_emotion', ''),
                        'emotion_confidence': emotions.get('confidence', 0),
                    })
                    
                    # Add individual emotion scores
                    all_emotions = emotions.get('all_emotions', {})
                    for emotion, score in all_emotions.items():
                        flat_row[f'emotion_{emotion}'] = score
                
                # Flatten posture data
                if 'posture' in row and row['posture']:
                    posture = row['posture']
                    flat_row.update({
                        'posture_type': posture.get('posture_type', ''),
                        'posture_confidence': posture.get('confidence', 0),
                    })
                    
                    # Add pose analysis data
                    pose_analysis = posture.get('pose_analysis', {})
                    for key, value in pose_analysis.items():
                        flat_row[f'posture_{key}'] = value
                
                flattened_data.append(flat_row)
            
            export_df = pd.DataFrame(flattened_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            logging.error(f"Error exporting data: {str(e)}")
