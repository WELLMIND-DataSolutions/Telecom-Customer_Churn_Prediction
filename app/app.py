# app.py - Single Customer Churn Prediction with Shape (No Threshold)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SHAPE - Data Structure (Mana + Shape Mix)
# ============================================

@dataclass
class CustomerShape:
    """Shape / Schema for Customer Data - NO VALIDATION"""
    # Required fields - koi bhi value le sakte hain
    tenure_months: int
    monthly_charges: float
    support_tickets: int
    satisfaction_score: int
    contract_type: str
    late_payments: int
    
    # Optional fields with defaults
    total_charges: float = 0.0
    payment_method: str = "Electronic"
    auto_pay: int = 0
    paperless_billing: int = 0
    num_services: int = 1
    
    # NO __post_init__ validation - koi check nahi
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert shape to dictionary"""
        return {
            'tenure_months': self.tenure_months,
            'monthly_charges': self.monthly_charges,
            'total_charges': self.total_charges,
            'support_tickets': self.support_tickets,
            'satisfaction_score': self.satisfaction_score,
            'late_payments': self.late_payments,
            'num_services': self.num_services,
            'contract_type': self.contract_type,
            'payment_method': self.payment_method,
            'auto_pay': self.auto_pay,
            'paperless_billing': self.paperless_billing
        }


@dataclass
class PredictionResultShape:
    """Shape for Prediction Result"""
    risk_score: float
    risk_level: str
    risk_color: str
    risk_icon: str
    message: str
    risk_factors: List[str]
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'risk_color': self.risk_color,
            'risk_icon': self.risk_icon,
            'message': self.message,
            'risk_factors': self.risk_factors,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }


# ============================================
# BACKEND LOGIC (Mana - Prediction Engine)
# ============================================

class ChurnPredictionEngine:
    """Mana - Backend prediction logic - NO THRESHOLD"""
    
    def __init__(self):
        self.weights = {
            'short_tenure': 0.25,
            'high_charges': 0.15,
            'high_tickets': 0.20,
            'low_satisfaction': 0.15,
            'late_payments': 0.15,
            'month_to_month': 0.10
        }
    
    def predict(self, customer: CustomerShape) -> PredictionResultShape:
        """Calculate churn risk based on customer shape - koi limit nahi"""
        risk_score = 0.0
        risk_factors = []
        
        # Rule 1: Short tenure - koi bhi tenure ho sakta hai
        if customer.tenure_months < 12:
            risk_score += self.weights['short_tenure']
            risk_factors.append(f"Short tenure ({customer.tenure_months} months)")
        elif customer.tenure_months > 60:
            # Extra safe for loyal customers
            pass
        
        # Rule 2: High monthly charges - koi bhi amount
        if customer.monthly_charges > 100:
            risk_score += self.weights['high_charges']
            risk_factors.append(f"High monthly charges (${customer.monthly_charges:.0f})")
        
        # Rule 3: Multiple support tickets - koi bhi ticket count
        if customer.support_tickets > 3:
            risk_score += self.weights['high_tickets']
            risk_factors.append(f"{customer.support_tickets} support tickets")
        elif customer.support_tickets > 10:
            # Extra risk for very high tickets
            risk_score += 0.10
            risk_factors.append(f"Very high support tickets ({customer.support_tickets})")
        
        # Rule 4: Low satisfaction - 1 se 5 ke beech mein kuch bhi
        if customer.satisfaction_score < 3:
            risk_score += self.weights['low_satisfaction']
            risk_factors.append(f"Low satisfaction score ({customer.satisfaction_score}/5)")
        
        # Rule 5: Late payments - koi bhi count
        if customer.late_payments > 2:
            risk_score += self.weights['late_payments']
            risk_factors.append(f"{customer.late_payments} late payments")
        
        # Rule 6: Month-to-month contract
        if customer.contract_type == "Month-to-month":
            risk_score += self.weights['month_to_month']
            risk_factors.append("Month-to-month contract (higher risk)")
        
        # Risk score ko cap karna (0 se 1 ke beech)
        risk_score = max(0.0, min(risk_score, 0.99))
        
        # Determine risk level - based on final score
        if risk_score >= 0.7:
            risk_level = "HIGH"
            risk_color = "churn-high"
            risk_icon = "🔴"
            message = "⚠️ Immediate intervention needed! High probability of churn."
            recommendations = [
                "📞 Schedule immediate customer support call",
                "💰 Offer loyalty discount (20-30%)",
                "⭐ Send satisfaction survey",
                "🔄 Offer contract upgrade with benefits",
                "👤 Assign dedicated account manager"
            ]
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
            risk_color = "churn-medium"
            risk_icon = "🟡"
            message = "⚠️ Monitor closely and engage proactively"
            recommendations = [
                "📧 Send engagement email campaign",
                "🎁 Offer small incentive or discount",
                "📞 Check-in call in next 30 days",
                "⭐ Request feedback",
                "📱 Send app usage tips"
            ]
        else:
            risk_level = "LOW"
            risk_color = "churn-low"
            risk_icon = "🟢"
            message = "✅ Customer is likely to stay. Focus on retention."
            recommendations = [
                "🎯 Offer cross-sell opportunities",
                "⭐ Request positive reviews/referrals",
                "🎁 Loyalty program benefits",
                "📧 Send personalized appreciation",
                "🔔 Early access to new features"
            ]
        
        return PredictionResultShape(
            risk_score=risk_score,
            risk_level=risk_level,
            risk_color=risk_color,
            risk_icon=risk_icon,
            message=message,
            risk_factors=risk_factors,
            recommendations=recommendations
        )


# ============================================
# FRONTEND - Streamlit UI
# ============================================

# Page configuration
st.set_page_config(
    page_title="Churn Prediction - Single Customer",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .churn-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .churn-medium {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: white;
    }
    .churn-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-factor {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    hr {
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔮 Customer Churn Prediction</h1>
    <p>Single Customer Risk Assessment - No Limits</p>
</div>
""", unsafe_allow_html=True)

# Initialize prediction engine
@st.cache_resource
def get_engine():
    return ChurnPredictionEngine()

engine = get_engine()

# Input Form - NO THRESHOLD, koi bhi value daal sakte hain
st.markdown("### 📋 Enter Customer Details (Any Values Allowed)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📅 Account Information**")
    tenure_months = st.number_input("Tenure (Months)", value=12, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", value=70.0, step=5.0)
    total_charges = st.number_input("Total Charges ($)", value=840.0, step=50.0)
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
with col2:
    st.markdown("**📊 Service & Support**")
    support_tickets = st.number_input("Support Tickets (last 6 months)", value=2, step=1)
    satisfaction_score = st.slider("Customer Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
    late_payments = st.number_input("Late Payments (last 12 months)", value=0, step=1)
    payment_method = st.selectbox("Payment Method", ["Electronic", "Mailed", "Bank transfer", "Credit card"])

# Optional Details (Collapsed)
with st.expander("🔧 Additional Details (Optional)"):
    col1, col2 = st.columns(2)
    with col1:
        auto_pay = st.selectbox("Auto-pay Enabled", [0, 1], format_func=lambda x: "Yes" if x else "No")
        num_services = st.slider("Number of Services", min_value=1, max_value=20, value=3)
    with col2:
        paperless_billing = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Predict Button
st.markdown("---")
predict_button = st.button("🚀 Predict Churn Risk", use_container_width=True, type="primary")

# Prediction Result
if predict_button:
    try:
        # Create Customer Shape - NO VALIDATION, jo bhi daaloge accept hoga
        customer = CustomerShape(
            tenure_months=int(tenure_months),
            monthly_charges=float(monthly_charges),
            total_charges=float(total_charges),
            support_tickets=int(support_tickets),
            satisfaction_score=int(satisfaction_score),
            contract_type=contract_type,
            late_payments=int(late_payments),
            payment_method=payment_method,
            auto_pay=auto_pay,
            paperless_billing=paperless_billing,
            num_services=num_services
        )
        
        # Get prediction
        result = engine.predict(customer)
        
        # Display Result Card
        st.markdown("---")
        st.markdown("## 📊 Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="prediction-card {result.risk_color}">
                <h2>{result.risk_icon} {result.risk_level} CHURN RISK {result.risk_icon}</h2>
                <h1 style="font-size: 4rem;">{result.risk_score:.1%}</h1>
                <p>Probability of customer churning</p>
                <p><b>{result.message}</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Factors
        if result.risk_factors:
            st.markdown("### ⚠️ Risk Factors Detected")
            for factor in result.risk_factors:
                st.markdown(f'<div class="risk-factor">• {factor}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ No major risk factors detected! Customer appears satisfied.")
        
        # Recommendations
        st.markdown("### 💡 Recommended Actions")
        for rec in result.recommendations:
            st.markdown(f"- {rec}")
        
        # Summary Card
        st.markdown("---")
        st.markdown("### 📋 Prediction Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Score", f"{result.risk_score:.1%}")
            st.metric("Risk Level", result.risk_level)
            # Show raw input values
            st.metric("Tenure", f"{customer.tenure_months} months")
            st.metric("Monthly Charges", f"${customer.monthly_charges:.0f}")
        with col2:
            st.metric("Factors Count", len(result.risk_factors))
            st.metric("Predicted At", result.timestamp)
            st.metric("Support Tickets", customer.support_tickets)
            st.metric("Late Payments", customer.late_payments)
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Koi bhi value daal sakte ho - koi limit nahi hai!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>🏢 Customer Churn Prediction | Single Customer Assessment</p>
    <p>💡 No limits - aap kuch bhi value enter kar sakte hain (negative, zero, ya bahut bada number)</p>
</div>
""", unsafe_allow_html=True)