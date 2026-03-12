# import streamlit as st
# import os
# import pandas as pd
# import plotly.express as px

# # Import the detector functions you created from your detector.py file
# from detector import predict_with_facexray, predict_with_lipforensics, predict_with_efficientnet

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Deepfake Detection System",
#     page_icon="🤖",
#     layout="wide"
# )

# # --- Main App Interface ---
# st.title("Advanced Deepfake Detection System 🕵️")
# st.write(
#     "Upload a video to analyze it with a suite of advanced, open-source neural models. "
#     "The system will provide a probability score for each model, indicating the likelihood of manipulation."
# )

# # --- Model Paths ---
# # Define the paths to your models here to keep the main logic clean.
# # !!! IMPORTANT: Make sure these paths are correct for your system. !!!
# FACE_XRAY_MODEL_PATH = 'C:/Deepfake/facexray_model/best_model.pth.tar'
# HRNET_CONFIG_PATH = 'C:/Deepfake\HRNet/hrnet_config\experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
# LIP_FORENSICS_MODEL_PATH = 'C:/Deepfake/lipforensics_model/lipforensics_ff.pth'
# EFFICIENTNET_MODEL_PATH = 'C:/Deepfake\efficientnet_model/weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36'


# # --- File Uploader ---
# uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

# if uploaded_file is not None:
#     # Create a temporary directory to save the uploaded file
#     temp_dir = "temp"
#     os.makedirs(temp_dir, exist_ok=True)
#     video_path = os.path.join(temp_dir, uploaded_file.name)

#     # Save the uploaded file to the temporary path
#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # --- Display Video and Run Analysis ---
#     st.video(video_path)

#     if st.button("Analyze Video"):
#         scores = {}
#         with st.spinner("Analyzing video... This may take a few minutes depending on video length."):
            
#             # --- Run Each Detector ---
#             st.info("Running Detector 1: Face X-ray...")
#             scores['Face X-ray'] = predict_with_facexray(video_path, FACE_XRAY_MODEL_PATH, HRNET_CONFIG_PATH)
            
#             st.info("Running Detector 2: LipForensics...")
#             scores['LipForensics'] = predict_with_lipforensics(video_path, LIP_FORENSICS_MODEL_PATH)

#             st.info("Running Detector 3: EfficientNet B7...")
#             scores['EfficientNet B7'] = predict_with_efficientnet(video_path, EFFICIENTNET_MODEL_PATH)

#         st.success("Analysis Complete!")

#         # --- Display Results ---
#         st.header("Detection Results")

#         # --- UPDATED LOGIC: Decision is now based only on EfficientNet B7 score ---
#         efficientnet_score = scores['EfficientNet B7']
        
#         # Display the final verdict based on the 10% threshold
#         if efficientnet_score < 0.10:
#             st.success(f"**Final Verdict: REAL** ")
#             st.info("The analysis suggests this video is likely authentic.")
#         else:
#             st.error(f"**Final Verdict: FAKE** ")
#             st.warning("The analysis suggests a high probability that this video is a deepfake.")

#         # --- Detailed Scores and Visualization (Still shows all three models for context) ---
#         st.subheader("Individual Model Scores")
        
#         # Create a DataFrame for plotting
#         df_scores = pd.DataFrame(list(scores.items()), columns=['Model', 'Fake Probability'])

#         # Bar chart of individual scores
#         fig = px.bar(
#             df_scores, 
#             x='Model', 
#             y='Fake Probability', 
#             title='Fake Probability per Model',
#             color='Model',
#             range_y=[0, 1],
#             text_auto='.2%'
#         )
#         fig.update_traces(textposition='outside')
#         st.plotly_chart(fig, use_container_width=True)

#         # --- CSV Report Download ---
#         st.subheader("Download Report")
#         df_report = df_scores.copy()
        
#         # Add the final verdict to the report
#         final_verdict = "REAL" if efficientnet_score < 0.10 else "FAKE"
#         df_report.loc['Verdict'] = ['Final Verdict', final_verdict]

#         # Convert DataFrame to CSV for downloading
#         @st.cache_data
#         def convert_df_to_csv(df):
#             return df.to_csv(index=False).encode('utf-8')

#         csv = convert_df_to_csv(df_report)

#         st.download_button(
#             label="Download Results as CSV",
#             data=csv,
#             file_name=f"detection_report_{uploaded_file.name}.csv",
#             mime="text/csv",
#         )

#         # Clean up the temporary file
#         os.remove(video_path)

# import streamlit as st
# import os
# import pandas as pd
# import plotly.express as px

# # Import the detector functions you created from your detector.py file
# from detector import predict_with_facexray, predict_with_lipforensics, predict_with_efficientnet

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Deepfake Detection System",
#     page_icon="🤖",
#     layout="wide"
# )

# # --- Main App Interface ---
# st.title("Advanced Deepfake Detection System 🕵️")
# st.write(
#     "Upload a video to analyze it with a suite of advanced, open-source neural models. "
#     "The system will provide a probability score for each model, indicating the likelihood of manipulation."
# )

# # --- Model Paths ---
# FACE_XRAY_MODEL_PATH = 'C:/Deepfake/facexray_model/best_model.pth.tar'
# HRNET_CONFIG_PATH = 'C:/Deepfake/HRNet/hrnet_config/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
# LIP_FORENSICS_MODEL_PATH = 'C:/Deepfake/lipforensics_model/lipforensics_ff.pth'
# EFFICIENTNET_MODEL_PATH = 'C:/Deepfake/efficientnet_model/weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36'

# # --- File Uploader ---
# uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

# if uploaded_file is not None:
#     temp_dir = "temp"
#     os.makedirs(temp_dir, exist_ok=True)
#     video_path = os.path.join(temp_dir, uploaded_file.name)

#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     st.video(video_path)

#     if st.button("Analyze Video"):
#         scores = {}
#         with st.spinner("Analyzing video... This may take a few minutes depending on video length."):

#             # --- Run Each Detector (without showing running messages) ---
#             scores['Face X-ray'] = predict_with_facexray(video_path, FACE_XRAY_MODEL_PATH, HRNET_CONFIG_PATH)
#             scores['LipForensics'] = predict_with_lipforensics(video_path, LIP_FORENSICS_MODEL_PATH)
#             scores['EfficientNet B7'] = predict_with_efficientnet(video_path, EFFICIENTNET_MODEL_PATH)

#         st.success("Analysis Complete!")

#         # --- Display Results ---
#         st.header("Detection Results")

#         efficientnet_score = scores['EfficientNet B7']

#         # Display the final verdict prominently
#         if efficientnet_score < 0.10:
#             st.markdown("<b><h1 style='color:green;'>✅ Final Verdict: REAL</h1><b>", unsafe_allow_html=True)
#             st.info("The analysis suggests this video is likely authentic.")
#         else:
#             st.markdown("<h2 style='color:red;'>❌ Final Verdict: FAKE</h2>", unsafe_allow_html=True)
#             st.warning("The analysis suggests a high probability that this video is a deepfake.")

#         # --- Detailed Scores ---
#         st.subheader("Individual Model Scores")
#         df_scores = pd.DataFrame(list(scores.items()), columns=['Model', 'Fake Probability'])

#         fig = px.bar(
#             df_scores, 
#             x='Model', 
#             y='Fake Probability', 
#             title='Fake Probability per Model',
#             color='Model',
#             range_y=[0, 1],
#             text_auto='.2%'
#         )
#         fig.update_traces(textposition='outside')
#         st.plotly_chart(fig, use_container_width=True)

#         # --- CSV Report ---
#         st.subheader("Download Report")
#         df_report = df_scores.copy()
#         final_verdict = "REAL" if efficientnet_score < 0.10 else "FAKE"
#         df_report.loc['Verdict'] = ['Final Verdict', final_verdict]

#         @st.cache_data
#         def convert_df_to_csv(df):
#             return df.to_csv(index=False).encode('utf-8')

#         csv = convert_df_to_csv(df_report)

#         st.download_button(
#             label="Download Results as CSV",
#             data=csv,
#             file_name=f"detection_report_{uploaded_file.name}.csv",
#             mime="text/csv",
#         )

#         os.remove(video_path)




# import streamlit as st
# import os
# import pandas as pd
# import plotly.express as px

# # Import the detector functions you created from your detector.py file
# from detector import predict_with_facexray, predict_with_lipforensics, predict_with_efficientnet

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Deepfake Detection System",
#     page_icon="🤖",
#     layout="wide"
# )

# # --- Main App Interface ---
# st.title("Advanced Deepfake Detection System 🕵️")
# st.write(
#     "Upload a video to analyze it with a suite of advanced, open-source neural models. "
#     "The system will provide a probability score for each model, indicating the likelihood of manipulation."
# )

# # --- Model Paths ---
# FACE_XRAY_MODEL_PATH = 'C:/Deepfake/facexray_model/best_model.pth.tar'
# HRNET_CONFIG_PATH = 'C:/Deepfake/HRNet/hrnet_config/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
# LIP_FORENSICS_MODEL_PATH = 'C:/Deepfake/lipforensics_model/lipforensics_ff.pth'
# EFFICIENTNET_MODEL_PATH = 'C:/Deepfake/efficientnet_model/weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36'

# # --- File Uploader ---
# uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

# if uploaded_file is not None:
#     temp_dir = "temp"
#     os.makedirs(temp_dir, exist_ok=True)
#     video_path = os.path.join(temp_dir, uploaded_file.name)

#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # --- Display video in center with small width ---
#     col1, col2, col3 = st.columns([1, 2, 1])  # middle col wider
#     with col2:
#         st.video(video_path, format="video/mp4", start_time=0)

#     if st.button("Analyze Video"):
#         scores = {}
#         with st.spinner("Analyzing video... This may take a few minutes depending on video length."):

#             # --- Run Each Detector (no running messages) ---
#             scores['Face X-ray'] = predict_with_facexray(video_path, FACE_XRAY_MODEL_PATH, HRNET_CONFIG_PATH)
#             scores





import streamlit as st
import os
import pandas as pd
import plotly.express as px

# Import the detector functions you created from your detector.py file
from detector import predict_with_facexray, predict_with_lipforensics, predict_with_efficientnet

# --- Page Configuration ---
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🤖",
    layout="wide"
)

# --- Main App Interface ---
st.title("Advanced Deepfake Detection System 🕵️")
st.write(
    "Upload a video to analyze it with a suite of advanced, open-source neural models. "
    "The system will provide a probability score for each model, indicating the likelihood of manipulation."
)

# --- Model Paths ---
FACE_XRAY_MODEL_PATH = 'C:/Deepfake/facexray_model/best_model.pth.tar'
HRNET_CONFIG_PATH = 'C:/Deepfake/HRNet/hrnet_config/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
LIP_FORENSICS_MODEL_PATH = 'C:/Deepfake/lipforensics_model/lipforensics_ff.pth'
EFFICIENTNET_MODEL_PATH = 'C:/Deepfake/efficientnet_model/weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36'

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, uploaded_file.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- Display video smaller and centered ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.video(video_path)

    if st.button("Analyze Video"):
        scores = {}
        with st.spinner("Analyzing video... This may take a few minutes depending on video length."):

            # --- Run Each Detector (without showing running messages) ---
            scores['Face X-ray'] = predict_with_facexray(video_path, FACE_XRAY_MODEL_PATH, HRNET_CONFIG_PATH)
            scores['LipForensics'] = predict_with_lipforensics(video_path, LIP_FORENSICS_MODEL_PATH)
            scores['EfficientNet B7'] = predict_with_efficientnet(video_path, EFFICIENTNET_MODEL_PATH)

        st.success("Analysis Complete!")

        # --- Display Results ---
        st.header("Detection Results")

        efficientnet_score = scores['EfficientNet B7']

        # Display the final verdict prominently and centered
        if efficientnet_score < 0.10:
            st.markdown("<h1 style='color:green; text-align:center;'>✅ Final Verdict: REAL</h1>", unsafe_allow_html=True)
            st.info("The analysis suggests this video is likely authentic.")
        else:
            st.markdown("<h1 style='color:red; text-align:center;'>❌ Final Verdict: FAKE</h1>", unsafe_allow_html=True)
            st.warning("The analysis suggests a high probability that this video is a deepfake.")

        # --- Detailed Scores ---
        st.subheader("Individual Model Scores")
        df_scores = pd.DataFrame(list(scores.items()), columns=['Model', 'Fake Probability'])

        fig = px.bar(
            df_scores, 
            x='Model', 
            y='Fake Probability', 
            title='Fake Probability per Model',
            color='Model',
            range_y=[0, 1],
            text_auto='.2%'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # --- CSV Report ---
        st.subheader("Download Report")
        df_report = df_scores.copy()
        final_verdict = "REAL" if efficientnet_score < 0.10 else "FAKE"
        df_report.loc['Verdict'] = ['Final Verdict', final_verdict]

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(df_report)

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"detection_report_{uploaded_file.name}.csv",
            mime="text/csv",
        )

        os.remove(video_path)