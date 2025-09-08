import pandas as pd
import torch
import config
from dataset import LoanApprovalDataset
from model import LoanApprovalClassificationModel

def main():
    # -------------------------
    # Select device
    # -------------------------
    print("-------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Selected device: {device}")

    # -------------------------
    # Load dataset normalization params and categorical mappings
    # -------------------------
    dataset = LoanApprovalDataset()
    dataset.load_feature_transformer()

    # -------------------------
    # Example real-world input
    # -------------------------
    df = pd.DataFrame([
        {
            "person_age": 22.0, "person_gender": "female", "person_education": "Master",
            "person_income": 71948.0, "person_emp_exp": 0, "person_home_ownership": "RENT",
            "loan_amnt": 35000.0, "loan_intent": "PERSONAL", "loan_int_rate": 16.02,
            "loan_percent_income": 0.49, "cb_person_cred_hist_length": 3.0, "credit_score": 561,
            "previous_loan_defaults_on_file": "No", "loan_status": 1
        },
        {
            "person_age": 21.0, "person_gender": "female", "person_education": "High School",
            "person_income": 12282.0, "person_emp_exp": 0, "person_home_ownership": "OWN",
            "loan_amnt": 1000.0, "loan_intent": "EDUCATION", "loan_int_rate": 11.14,
            "loan_percent_income": 0.08, "cb_person_cred_hist_length": 2.0, "credit_score": 504,
            "previous_loan_defaults_on_file": "Yes", "loan_status": 0
        },
        {
            "person_age": 25.0, "person_gender": "female", "person_education": "High School",
            "person_income": 12438.0, "person_emp_exp": 3, "person_home_ownership": "MORTGAGE",
            "loan_amnt": 5500.0, "loan_intent": "MEDICAL", "loan_int_rate": 12.87,
            "loan_percent_income": 0.44, "cb_person_cred_hist_length": 3.0, "credit_score": 635,
            "previous_loan_defaults_on_file": "No", "loan_status": 1
        },
        {
            "person_age": 23.0, "person_gender": "female", "person_education": "Bachelor",
            "person_income": 79753.0, "person_emp_exp": 0, "person_home_ownership": "RENT",
            "loan_amnt": 35000.0, "loan_intent": "MEDICAL", "loan_int_rate": 15.23,
            "loan_percent_income": 0.44, "cb_person_cred_hist_length": 2.0, "credit_score": 675,
            "previous_loan_defaults_on_file": "No", "loan_status": 1
        },
        {
            "person_age": 24.0, "person_gender": "male", "person_education": "Master",
            "person_income": 66135.0, "person_emp_exp": 1, "person_home_ownership": "RENT",
            "loan_amnt": 35000.0, "loan_intent": "MEDICAL", "loan_int_rate": 14.27,
            "loan_percent_income": 0.53, "cb_person_cred_hist_length": 4.0, "credit_score": 586,
            "previous_loan_defaults_on_file": "No", "loan_status": 1
        }
    ])

    X = dataset.prepare_data_for_inference(df)
    input_dim = X[0].numel()

    # -------------------------
    # Load trained model
    # -------------------------
    model = LoanApprovalClassificationModel(input_dim=input_dim, device=device)
    model.load()

    # -------------------------
    # Model inference
    # -------------------------
    model.eval()
    X = X.to(device)
    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > config.CLASSIFICATION_THRESHOLD).float()

    # -------------------------
    # Display predictions
    # -------------------------
    print("Predicted Loan Approvals:")
    for idx, row in df.iterrows():
        status = "Approved" if predictions[idx].item() == 1.0 else "Rejected"
        prob = probabilities[idx].item()
        
        print(f"----------------------------")
        print(f"Applicant {idx + 1}")
        print(f"----------------------------")
        print(f"Age: {row['person_age']}")
        print(f"Gender: {row['person_gender']}")
        print(f"Education: {row['person_education']}")
        print(f"Income: ${row['person_income']:,.2f}")
        print(f"Employment Experience: {row['person_emp_exp']} years")
        print(f"Home Ownership: {row['person_home_ownership']}")
        print(f"Loan Amount Requested: ${row['loan_amnt']:,.2f}")
        print(f"Loan Intent: {row['loan_intent']}")
        print(f"Loan Interest Rate: {row['loan_int_rate']}%")
        print(f"Loan % of Income: {row['loan_percent_income']*100:.1f}%")
        print(f"Credit History Length: {row['cb_person_cred_hist_length']} years")
        print(f"Credit Score: {row['credit_score']}")
        print(f"Previous Loan Defaults: {row['previous_loan_defaults_on_file']}")
        print(f"Predicted Loan Status: {status} (Probability: {prob:.2f})")
        print()

if __name__ == "__main__":
    main()
