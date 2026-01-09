import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# Load the model
model = joblib.load('xgb_model.pkl')

# Define feature names used for the model
feature_names = [
    "Hb", "PLT", "ALT", "BUN", "UA", "HDL"
]

# Streamlit user interface
st.title("Risk Prediction of Diabetic Nephropathy in Elderly Patients with Type 2 Diabetes in the Community")
st.title("社区老年二型糖尿病患者糖尿病肾病风险预测")

# Hb: numerical input
hb = st.number_input("Hb (Hemoglobin) (血红蛋白) <g/L>:", min_value=50, max_value=200, value=120)

# PLT: numerical input
Plt = st.number_input("PLT (Platelets) (血小板) <10^9/L>:", min_value=10, max_value=500, value=280)

# ALT: numerical input
alt = st.number_input("ALT (Alanine Aminotransferase) (血清谷丙转氨酶) <U/L>:",  value=25)

# BUN: numerical input
bun = st.number_input("BUN (Blood urea nitrogen) (血尿素氮) <mmol/L>:", min_value=0, max_value=50, value=5)

# UA: numerical input
ua = st.number_input("UA (Uric Acid) (尿酸) <μmol/L>:", min_value=100, max_value=800, value=350)

# HDL: numerical input
hdl = st.number_input("HDL (High-Density Lipoprotein Cholesterol) (高密度脂蛋白胆固醇) <mmol/L>:", value=2)


# Process inputs and make predictions
feature_values = [hb, Plt , alt, bun, ua, hdl]
features = np.array([feature_values], dtype=float)

if st.button("Predict (预测)"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class (预测类别):** {predicted_class}")
    st.write(f"**Prediction Probabilities (预测概率):** {predicted_proba}")

    # 根据鞠策结果生成建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            "According to this model, you may have a higher risk of developing diabetic nephropathy.\n"
            f"The predicted probability of developing diabetic nephropathy is {probability:.1f}%.\n"
            "It is recommended that you see a doctor as soon as possible for a more detailed diagnosis and appropriate treatment.\n\n"
            "根据模型预测，您可能存在较高的糖尿病肾病发病风险。\n"
            f"模型预测的糖尿病肾病发病概率为 {probability:.1f}%。\n"
            "建议您尽快就医，以进行更详细的诊断和采取适当的治疗措施。"
        )
    else:
        advice = (
            "According to the model, your risk of diabetic nephropathy is low.\n"
            f"The predicted probability of not having diabetic nephropathy is {probability:.1f}%.\n"
            "It is recommended that you maintain a healthy lifestyle and monitor your health regularly. If you experience any symptoms, please see a doctor promptly.\n\n"
            "根据模型预测，您的糖尿病肾病风险较低。\n"
            f"模型预测的无糖尿病肾病概率为 {probability:.1f}%。\n"
            "建议您继续保持健康的生活方式，并定期观察健康状况。如有任何异常症状，请及时就医。"
        )

    st.write(advice)
    
      # Calculate SHAP values and display force plot
X = pd.DataFrame([feature_values], columns=feature_names)

try:
    # 对 XGBClassifier 更稳：解释 booster
    explainer = shap.TreeExplainer(model.get_booster())

    shap_values = explainer.shap_values(X)

    # 二分类时 shap_values 可能是 list: [class0, class1]
    if isinstance(shap_values, list):
        shap_vec = shap_values[1][0]  # 取正类(1)的解释
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
    else:
        shap_vec = shap_values[0]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]

    plt.figure()
    shap.force_plot(base_value, shap_vec, X.iloc[0, :], matplotlib=True)
    st.pyplot(plt.gcf(), clear_figure=True)

except Exception as e:
    st.warning(f"SHAP plot could not be generated: {e}")

    # # 创建SHAP解释器，基于树模型
    # explainer = shap.TreeExplainer(model)
    # #计算shap值
    # shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    # st.image("shap_force_plot.png")





    



