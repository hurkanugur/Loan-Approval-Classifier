import torch
import gradio as gr
import pandas as pd
from src import config
from src.dataset import LoanApprovalDataset
from src.model import LoanApprovalClassificationModel

class InferencePipeline:
    """
    Handles model loading, preprocessing, and loan approval prediction.
    """

    # ----------------- Initialization -----------------
    
    def __init__(
            self, 
            model: LoanApprovalClassificationModel, 
            dataset: LoanApprovalDataset, 
            device: torch.device
        ):

        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()

    # ----------------- Public Methods -----------------

    def predict(self, features: dict) -> str:
        """
        Predict the car price given a dictionary of features.
        Returns a formatted string with the predicted price.
        """

        # Fill empty numeric values with 0
        for key, value in features.items():
            if value is None or (isinstance(value, str) and value.strip() == ""):
                features[key] = 0

        # Convert dict to single-row DataFrame
        df = pd.DataFrame([features])

        # Preprocess features for the model
        X = self.dataset.prepare_data_for_inference(df)
        X = X.to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.model(X)
            probabiliy = torch.sigmoid(outputs)
            prediction = (probabiliy > config.CLASSIFICATION_THRESHOLD).float()

        # Generate label
        if prediction == 0:
            confidence = round((1 - probabiliy.item()) * 100, 2)
            return f"❌ Rejected — {confidence}% confidence"
        elif prediction == 1:
            confidence = round(probabiliy.item() * 100, 2)
            return f"✅ Approved — {confidence}% confidence"
        else:
            return "⚠️ Unknown classification!"

    def create_gradio_app(self) -> gr.Blocks:
        """
        Build and return the Gradio interface for interactive inference.
        """
        with gr.Blocks(theme=gr.themes.Ocean(), title="Loan Approval Predictor") as app:
            gr.Markdown(
                """
                # 💰 Loan Approval Predictor  
                Enter applicant details to see if the loan is **likely to be approved**!
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Personal Information")
                    age = gr.Slider(18, 70, step=1, label="Age")
                    gender = gr.Dropdown(["male", "female"], label="Gender")
                    education = gr.Dropdown(
                        ["High School", "Bachelor", "Master", "Doctorate", "Other"],
                        label="Education Level"
                    )
                    income = gr.Number(label="Annual Income ($)", value=30000)

                    emp_exp = gr.Slider(0, 40, step=1, label="Employment Experience (years)")
                    home_ownership = gr.Dropdown(
                        ["RENT", "OWN", "MORTGAGE", "OTHER"],
                        label="Home Ownership"
                    )

                    with gr.Column(scale=1):
                        predict_btn = gr.Button("🔍 Predict Loan Status", variant="primary")
                        clear_btn = gr.Button("🧹 Clear", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("### 💳 Loan Details")
                    loan_amount = gr.Number(label="Loan Amount Requested ($)")
                    loan_intent = gr.Dropdown(
                        ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                        label="Loan Intent"
                    )
                    loan_int_rate = gr.Slider(1, 25, step=0.1, label="Interest Rate (%)")
                    loan_percent_income = gr.Slider(0.01, 1.0, step=0.01, label="Loan as % of Income")

                    credit_hist_len = gr.Slider(0, 50, step=1, label="Credit History Length (years)")
                    credit_score = gr.Slider(300, 850, step=1, label="Credit Score")
                    prev_defaults = gr.Dropdown(["Yes", "No"], label="Previous Loan Defaults")

                    with gr.Column(scale=1):
                        result = gr.Textbox(
                            label="Loan Prediction Result",
                            placeholder="Prediction will appear here...",
                            interactive=False,
                            lines=2,
                            show_copy_button=True,
                        )

            # Prediction logic
            predict_btn.click(
                fn=lambda a, g, e, i, ex, h, la, li, lir, lpi, chl, cs, pd: self.predict({
                    "person_age": a,
                    "person_gender": g,
                    "person_education": e,
                    "person_income": i,
                    "person_emp_exp": ex,
                    "person_home_ownership": h,
                    "loan_amnt": la,
                    "loan_intent": li,
                    "loan_int_rate": lir,
                    "loan_percent_income": lpi,
                    "cb_person_cred_hist_length": chl,
                    "credit_score": cs,
                    "previous_loan_defaults_on_file": pd
                }),
                inputs=[
                    age, gender, education, income, emp_exp, home_ownership,
                    loan_amount, loan_intent, loan_int_rate, loan_percent_income,
                    credit_hist_len, credit_score, prev_defaults
                ],
                outputs=result
            )

            clear_btn.click(
                fn=lambda: (
                    30, "male", "High School", 30000, 0, "RENT",
                    0, "PERSONAL", 5, 0.2, 0, 650, "No", ""
                ),
                inputs=None,
                outputs=[
                    age, gender, education, income, emp_exp, home_ownership,
                    loan_amount, loan_intent, loan_int_rate, loan_percent_income,
                    credit_hist_len, credit_score, prev_defaults, result
                ]
            )

            gr.Markdown(
                """
                ---
                💡 **Tip:**  
                - Higher income and credit score generally improve approval odds.  
                - A lower loan-to-income ratio and clean credit history help too.  

                ---
                👨‍💻 **Developed by [Hürkan Uğur](https://github.com/hurkanugur)**  
                🔗 Source Code: [Loan-Approval-Classifier](https://github.com/hurkanugur/Loan-Approval-Classifier)
                """
            )

        return app