import streamlit as st
import requests
import json

# Configuration
SERVER_URL = "http://127.0.0.1:8000"

def get_server_info():
    """Get server information and model status."""
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def predict_tennis(record):
    """Send prediction request to server."""
    try:
        response = requests.post(
            f"{SERVER_URL}/predict", 
            json={"record": record},
            headers={"Content-Type": "application/json"}
        )
        return response.json(), response.status_code == 200
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}, False

def main():
    st.set_page_config(
        page_title="Tennis Prediction Client",
        page_icon="🎾",
        layout="wide"
    )
    
    st.title("🎾 Tennis Game Predictor")
    st.markdown("---")
    
    # Check server status
    server_info = get_server_info()
    
    if server_info is None:
        st.error("❌ Cannot connect to the prediction server!")
        st.info("Make sure the server is running on http://127.0.0.1:8000")
        st.code("python src/server.py")
        return
    
    # Display server status and model info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Information")
        if server_info.get("model_status") == "Ready":
            st.success(f"✅ {server_info.get('model_status')}")
            accuracy = server_info.get('accuracy', 'Unknown')
            st.metric("Model Accuracy", accuracy)
        else:
            st.error(f"❌ {server_info.get('model_status')}")
            if 'error' in server_info:
                st.error(f"Error: {server_info['error']}")
    
    with col2:
        st.subheader("🌤️ Available Features")
        features = server_info.get('available_features', {})
        if features:
            for feature, values in features.items():
                st.write(f"**{feature}:** {', '.join(values)}")
    
    st.markdown("---")
    
    # Prediction form
    st.subheader("🔮 Make a Prediction")
    
    if features:
        # Create input form based on available features
        record = {}
        
        # Create columns for better layout
        input_cols = st.columns(len(features))
        
        for i, (feature, values) in enumerate(features.items()):
            with input_cols[i]:
                record[feature] = st.selectbox(
                    f"Select {feature}:",
                    options=values,
                    key=f"input_{feature}"
                )
        
        st.markdown("### Your Input:")
        st.json(record)
        
        # Prediction button
        if st.button("🎯 Predict Tennis Game", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                result, success = predict_tennis(record)
            
            if success:
                st.markdown("### 🎉 Prediction Result:")
                
                prediction = result.get('prediction')
                message = result.get('message')
                
                if prediction == 'Yes':
                    st.success(f"✅ {message}")
                else:
                    st.warning(f"⛔ {message}")
                
                # Show detailed results
                with st.expander("📋 Detailed Results"):
                    st.json(result)
                    
            else:
                st.error("❌ Prediction failed!")
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                if 'required_features' in result:
                    st.info("Required features:")
                    st.json(result['required_features'])
                
                # Show full error response
                with st.expander("🔍 Full Error Details"):
                    st.json(result)
    
    else:
        st.warning("⚠️ No features available from server")
    
    st.markdown("---")
    
    # Instructions section
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Instructions:
        1. **Check Model Status**: Make sure the model is ready and see its accuracy
        2. **Select Features**: Choose values for each weather condition:
           - **Outlook**: Weather outlook (Sunny, Overcast, Rain)
           - **Temperature**: Temperature level (Hot, Mild, Cool)
           - **Humidity**: Humidity level (High, Normal)
           - **Wind**: Wind condition (Strong, Weak)
        3. **Make Prediction**: Click the predict button to get the result
        4. **View Result**: See if tennis should be played based on weather conditions
        
        ### Example Scenarios:
        - **Good Tennis Weather**: Overcast, Mild, Normal, Weak → Usually "Yes"
        - **Bad Tennis Weather**: Rain, Hot, High, Strong → Usually "No"
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🎾 Tennis Prediction Client • Powered by Naive Bayes Classifier"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
