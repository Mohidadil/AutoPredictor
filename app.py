import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import json
import sqlite3
import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # Auto ARIMA ke liye
import joblib  # Model loading ke liye
import joblib  # Model saving ke liye
import pickle
import datetime

# Custom CSS for black background, golden text, and footer styling
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit style
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #1C2526;
    }
    /* All text to golden */
    .stApp, h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #FFD700 !important;
    }
    /* Widget labels and inputs */
    .stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label, .stFileUploader label {
        color: #FFD700 !important;
    }
    /* Dataframe text */
    .stDataFrame div, .stDataFrame span {
        color: #FFD700 !important;
    }
    /* Buttons */
    .stButton button {
        background-color: #FF4500;
        color: #FFD700;
    }
    /* Multiselect background */
    .stMultiSelect div {
        background-color: #2F4F4F !important;
    }
    /* Footer styling */
    .footer {
        background-color: #1C2526;
        padding: 20px;
        text-align: center;
        border-top: 2px solid #FFD700;
        margin-top: 50px;
    }
    .footer a {
        color: #FFD700;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        color: #FF4500;
    }
    .footer p {
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Matplotlib style
plt.rcParams.update({
    'text.color': '#FFD700',         # sab text golden
    'axes.labelcolor': '#FFD700',     # x, y labels golden
    'xtick.color': '#FFD700',         # x-axis ticks golden
    'ytick.color': '#FFD700',         # y-axis ticks golden
    'axes.titlecolor': '#FFD700',     # title golden
    'legend.edgecolor': '#FFD700',    # legend border golden
    'legend.labelcolor': '#FFD700',   # legend text golden
})


def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    elif file.name.endswith('.txt'):
        return pd.read_csv(file, delimiter="\t")
    elif file.name.endswith('.db'):
        conn = sqlite3.connect(file.name)
        query = "SELECT * FROM data"
        return pd.read_sql(query, conn)
    else:
        st.error("Unsupported file format!")
        return None


def preprocess_data(df, task):
    st.write("### Dataset Preview")
    st.dataframe(df.head(50))
    
    # 1. Missing Values Check
    st.write("### Missing Values in Each Column")
    st.dataframe(df.isnull().sum())
    
    # 2. Unique Values
    st.write("### Unique Values Per Column")
    unique_values = {col: df[col].unique().tolist() for col in df.columns}
    unique_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in unique_values.items()]))
    st.write("### Clean Dataset Preview")
    st.dataframe(df.head(30))
    st.dataframe(unique_df)
    
    # 3. Duplicates aur Empty Rows Hatao
    df.drop_duplicates(inplace=True)
    df.dropna(how='all', inplace=True)
    
    # 4. Data Profiling - Summary Report
    st.write("### Data Profiling - Summary Report")
    profile_data = {
        "Column": df.columns,
        "Min": [df[col].min() if df[col].dtype in ['int64', 'float64'] else 'N/A' for col in df.columns],
        "Max": [df[col].max() if df[col].dtype in ['int64', 'float64'] else 'N/A' for col in df.columns],
        "Mean": [df[col].mean() if df[col].dtype in ['int64', 'float64'] else 'N/A' for col in df.columns],
        "Median": [df[col].median() if df[col].dtype in ['int64', 'float64'] else 'N/A' for col in df.columns],
        "Std Dev": [df[col].std() if df[col].dtype in ['int64', 'float64'] else 'N/A' for col in df.columns],
        "Unique Count": [df[col].nunique() for col in df.columns],
        "Data Type": [df[col].dtype for col in df.columns]
    }
    profile_df = pd.DataFrame(profile_data)
    st.dataframe(profile_df)
    
    # 5. Custom Regex for Cleaning
    st.write("### Custom Regex for Special Characters")
    default_regex = r'[^A-Za-z0-9., ]'
    custom_regex = st.text_input("Enter custom regex pattern (leave blank for default)", default_regex)
    if not custom_regex:
        custom_regex = default_regex
    
    def clean_with_custom_regex(x):
        if isinstance(x, str):
            if re.match(r'^-?\d*\.?\d+$', x.replace(' ', '')):
                return float(x)
            numbers = re.findall(r'-?\d*\.?\d+', x)
            if numbers:
                return float(numbers[0])
            return re.sub(custom_regex, '', x)
        return x
    
    df = df.applymap(clean_with_custom_regex)
    
    # 6. String me Numbers Detect Karo aur Convert Karo
    def detect_and_convert_numeric(df):
        for col in df.columns:
            if df[col].dtype == 'object':
                all_numeric = df[col].apply(lambda x: bool(re.match(r'^-?\d*\.?\d+$', str(x)))).all()
                if all_numeric:
                    df[col] = pd.to_numeric(df[col])
        return df
    df = detect_and_convert_numeric(df)
    
    # 7. User se Column Types Chunwao
    def set_column_types(df):
        st.write("### Select Data Types for Columns")
        type_options = ['int', 'float', 'string', 'category', 'datetime']
        col_types = {}
        for col in df.columns:
            selected_type = st.selectbox(f"Select type for {col}", type_options, key=f"type_{col}")
            col_types[col] = selected_type
        
        for col, dtype in col_types.items():
            try:
                if dtype == 'int':
                    df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce').fillna(0).astype(int)
                elif dtype == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype == 'string':
                    df[col] = df[col].astype(str)
                elif dtype == 'category':
                    df[col] = df[col].astype('category')
                elif dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Error converting {col} to {dtype}: {e}")
        return df
    df = set_column_types(df)
    
    # 8. Missing Values Handle Karo (Naya Section)
    # 8. Missing Values Handle Karo (Modified Section)
    st.write("### Handle Missing Values for Each Column")

    # Check karo ke koi missing values hain ya nahi
    missing_values_exist = df.isnull().sum().sum() > 0

    if missing_values_exist:
        missing_strategies = {}
        strategy_options = ["Median", "Mean", "Mode", "Drop", "Constant"]

        # Har column ke liye strategy select karo
        for col in df.columns:
            if df[col].isnull().sum() > 0:  # Sirf un columns ke liye jo missing values hain
                st.write(f"**Column: {col}** (Missing Values: {df[col].isnull().sum()})")
                strategy = st.selectbox(
                    f"Select missing value strategy for {col}",
                    strategy_options,
                    key=f"missing_strategy_{col}"
                )
                missing_strategies[col] = strategy
                # Agar Constant strategy chuni, to user se constant value pooch lo
                if strategy == "Constant":
                    constant_value = st.text_input(
                        f"Enter constant value for {col}",
                        key=f"constant_value_{col}"
                    )
                    missing_strategies[col] = (strategy, constant_value)

        # Strategies apply karo
        for col, strategy in missing_strategies.items():
            try:
                if isinstance(strategy, tuple) and strategy[0] == "Constant":
                    strategy, constant_value = strategy
                    # Constant value ko column ke data type ke hisaab se convert karo
                    if df[col].dtype in ['int64', 'int32']:
                        df[col].fillna(int(constant_value), inplace=True)
                    elif df[col].dtype in ['float64', 'float32']:
                        df[col].fillna(float(constant_value), inplace=True)
                    else:
                        df[col].fillna(constant_value, inplace=True)
                elif strategy == "Median" and df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "Mean" and df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif strategy == "Drop":
                    df.dropna(subset=[col], inplace=True)
            except Exception as e:
                st.warning(f"Error applying {strategy} to {col}: {e}")
    else:
        st.info("No missing values found in the dataset. Skipping missing value handling.")
    
    # 9. Outliers Handle Karo
    outlier_action = st.radio("Handle outliers by:", ["Remove", "Cap"])
    for col in df.select_dtypes(include=['number']).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        if outlier_action == "Remove":
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        else:
            df[col] = df[col].clip(lower, upper)
    
    # 10. Features Select Karo
    st.write("### Selecting Important Features")
    target_column = st.selectbox("Select Target Column", df.columns, key="target_column_1")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if task in ["Classification", "Regression"]:
        model = RandomForestClassifier() if y.nunique() < 10 else RandomForestRegressor()
        model.fit(X, y)
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        selected_features = feature_importances[feature_importances > 0.01].index.tolist()
        df = df[selected_features + [target_column]]
        st.write("### Selected Features")
        st.dataframe(pd.DataFrame(feature_importances, columns=['Importance']))
    
    elif task == "Clustering":
        selector = VarianceThreshold(threshold=0.01)
        X_selected = selector.fit_transform(X)
        df = pd.DataFrame(X_selected, columns=X.columns[selector.get_support()])
        st.write("### Features Selected for Clustering")
        st.dataframe(df.head())
    
    elif task == "Time Series Analysis":
        st.write("### Auto-Correlation and Partial Auto-Correlation")
        acf_values = acf(y, nlags=20)
        pacf_values = pacf(y, nlags=20)
        df_acf_pacf = pd.DataFrame({"ACF": acf_values, "PACF": pacf_values})
        st.dataframe(df_acf_pacf)
    
    return df
    

@st.cache_resource  # Training ko cache karo

def encode_features(df, target_column):
    df_encoded = df.copy()
    label_encoders = {}

    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        if col == target_column:
            continue
        if df_encoded[col].nunique() <= 2:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
        else:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

    if df_encoded[target_column].dtype == 'object':
        le = LabelEncoder()
        df_encoded[target_column] = le.fit_transform(df_encoded[target_column])
        label_encoders[target_column] = le

    return df_encoded, label_encoders

def train_model(df, target_column):
    #import streamlit as st
    #import numpy as np
    #import pandas as pd
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    #import joblib
    #from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    #from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    #from sklearn.linear_model import LogisticRegression, LinearRegression
    #from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    #from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    #from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
    #from xgboost import XGBRegressor
    #from lightgbm import LGBMRegressor
#
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Problem Type ko detect karo
    if y.dtype == 'object' or y.nunique() <= 10:
        problem_type = 'classification'
    else:
        problem_type = 'regression'

    st.write(f"Detected Problem Type: **{problem_type}**")

    # 1. Check: Kam se kam 2 unique classes
    if y.nunique() < 2:
        st.error(f"❌ Target column '{target_column}' me sirf ek unique value ({y.unique()[0]}) hai. Model banana possible nahi.")
        return None, None

    # 2. Data ko train aur test me baanto
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type == 'classification' else None
    )

    # 3. Models define karo
    models = {
        'classification': {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        },
        'regression': {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor()
        }
    }

    # Baaki sab code wahi rahega
    st.write("### Select Models to Train")
    selected_models = st.multiselect(
        f"Choose {problem_type} models",
        list(models[problem_type].keys()),
        default=list(models[problem_type].keys())
    )
    models_to_train = {name: models[problem_type][name] for name in selected_models}

    scaler_options = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    scaler_choice = st.selectbox("Select Scaling Method", list(scaler_options.keys()), key="scaler_choice")
    scaler = scaler_options[scaler_choice]

    best_model, best_score = None, float('-inf')
    cv_details = {}
    for name, model in models_to_train.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        score = np.mean(scores)
        cv_details[name] = scores
        st.write(f"{name} Cross-Validation Mean Score: {score:.4f}")
        if score > best_score:
            best_score, best_model = score, model

    st.write("### Cross-Validation Fold Scores")
    for name, scores in cv_details.items():
        st.write(f"{name}:")
        st.write(f"Fold Scores: {[f'{s:.4f}' for s in scores]}")
        st.write(f"Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")

    param_grids = {
        'RandomForestClassifier': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'GradientBoostingClassifier': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        'LogisticRegression': {'C': [0.1, 1, 10], 'max_iter': [100, 200]},
        'DecisionTreeClassifier': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'RandomForestRegressor': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'GradientBoostingRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        'XGBRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1]},
        'LGBMRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1]},
        'DecisionTreeRegressor': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    }

    if best_model.__class__.__name__ in param_grids:
        st.write(f"Tuning hyperparameters for {best_model.__class__.__name__}")
        grid_search = GridSearchCV(best_model, param_grids[best_model.__class__.__name__], cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        st.write(f"Best Parameters: {grid_search.best_params_}")

    st.write("### Feature Importance Threshold")
    importance_threshold = st.slider(
        "Select Feature Importance Threshold",
        0.0, 0.1, 0.003, step=0.001, key="importance_threshold"
    )

    selected_features = X_train.columns
    if hasattr(best_model, "feature_importances_"):
        best_model.fit(X_train, y_train)
        feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        selected_features = feature_importances[feature_importances > importance_threshold].index.tolist()
        if len(selected_features) == 0:
            selected_features = X_train.columns
        st.write("### Selected Important Features")
        st.dataframe(pd.DataFrame(feature_importances, columns=['Importance']))

    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    st.write("### Save Trained Model")
    if st.button("Save Model"):
        model_filename = f"{best_model.__class__.__name__}_{problem_type}.joblib"
        joblib.dump(best_model, model_filename)
        st.success(f"Model saved as {model_filename}")

    if problem_type == 'classification':
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)

    else:
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
        st.write(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        st.pyplot(fig)

    return best_model, best_score

#@st.cache_data
# Yeh function widgets handle karega (non-cached)
def clustering_ui(df, clusters):
    st.write("### Clustering Options")
    clustering_method = st.selectbox("Select Clustering Method", ["KMeans", "DBSCAN", "Agglomerative"], key="clustering_method")
    
    if clustering_method == "KMeans":
        if clusters is None:
            clusters = st.slider("Select number of clusters", 2, 10, 3, key="kmeans_clusters")
        model = KMeans(n_clusters=clusters)
    elif clustering_method == "DBSCAN":
        eps = st.slider("Select epsilon (eps)", 0.1, 2.0, 0.5, step=0.1, key="dbscan_eps")
        min_samples = st.slider("Select min samples", 2, 10, 5, key="dbscan_min_samples")
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:  # Agglomerative
        if clusters is None:
            clusters = st.slider("Select number of clusters", 2, 10, 3, key="agglo_clusters")
        model = AgglomerativeClustering(n_clusters=clusters)

    show_elbow = clustering_method == "KMeans" and st.checkbox("Show Elbow Plot")
    show_viz = st.checkbox("Show Cluster Visualization") and len(df.select_dtypes(include=['number']).columns) >= 2

    return clustering_method, model, clusters, show_elbow, show_viz

# Yeh function clustering ka core kaam karega (cached)
@st.cache_data
def perform_clustering(df, _model):
    numeric_df = df.select_dtypes(include=['number'])
    if len(numeric_df.columns) < 1:
        raise ValueError("No numeric columns available for clustering.")
    df['Cluster'] = _model.fit_predict(numeric_df)
    return df, numeric_df

@st.cache_data
def time_series_analysis(df, column):
    # Check karo ke data time series ke liye tayyar hai
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        st.error("⚠️ Data ka index datetime hona chahiye time series ke liye. Pehle index set karo.")
        return None

    st.write("### Time Series Options")
    steps = st.slider("Select forecast steps", 5, 50, 10, key="forecast_steps")
    seasonal = st.checkbox("Use Seasonal ARIMA", key="seasonal_check")

    # Auto ARIMA model
    model = auto_arima(df[column], seasonal=seasonal, m=12 if seasonal else 1, trace=True, error_action='ignore')
    model_fit = model.fit(df[column])
    forecast = model_fit.predict(n_periods=steps)

    # Visualization
    st.write("### Time Series Forecast")
    forecast_index = pd.date_range(start=df.index[-1], periods=steps + 1, freq=df.index.inferred_freq)[1:]
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)
    combined_df = pd.concat([df[column], forecast_df], axis=1)

    fig = px.line(combined_df, title="Time Series Forecast")
    fig.add_scatter(x=df.index, y=df[column], mode='lines', name='Actual')
    fig.add_scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast')
    st.plotly_chart(fig)

    return forecast

def main():
    st.title("AutoPredictor Web")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json", "txt", "db"])
    
    if uploaded_file:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
        if df is not None:
            st.write("### Dataset Preview")
            st.dataframe(df.head())
            
            tasks = st.multiselect("Select tasks", ["Data Cleaning", "Visualization", "Model Training", "Clustering", "Time Series Analysis", "Regression", "Classification", "Underfitting", "Overfitting"])
            
            # Task dependency check
            if ("Model Training" in tasks or "Clustering" in tasks or "Regression" in tasks or "Classification" in tasks) and "Data Cleaning" not in tasks:
                st.warning("⚠️ Model Training ya Clustering ke liye pehle Data Cleaning chuno.")

            if "Data Cleaning" in tasks:
                with st.spinner("Cleaning data..."):
                    df = preprocess_data(df, tasks)
                st.write("Data cleaned successfully!")
                st.dataframe(df.head())
                
                # Download cleaned data
                csv = df.to_csv(index=False)
                st.download_button("Download Cleaned Data", csv, "cleaned_data.csv", "text/csv")

            if "Visualization" in tasks:
                st.write("### Data Visualizations")
                numeric_cols = df.select_dtypes(include=['number']).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns

                selected_columns = st.multiselect("Select Columns for Visualization", numeric_cols)
                visualization_type = st.radio("Select Visualization Library", ["Matplotlib", "Seaborn", "Plotly"])

                if len(selected_columns) > 0:
                    # Correlation Heatmap
                    if st.checkbox("Show Correlation Heatmap"):
                        plt.figure(figsize=(10,5), facecolor='#1C2526')
                        if visualization_type == "Seaborn":
                            sns.heatmap(df[selected_columns].corr(), annot=True, cmap='rainbow')
                        else:
                            fig = px.imshow(df[selected_columns].corr(), text_auto=True, aspect="auto", color_continuous_scale='Viridis')
                            st.plotly_chart(fig)
                        st.pyplot(plt)
                        plt.clf()

                    # Bar Plot (Grouped and Colorful)
                    if st.checkbox("Show Bar Plot"):
                        if len(categorical_cols) > 0:
                            group_col = st.selectbox("Select a categorical column for grouping (e.g., Region)", categorical_cols)
                            value_col = st.selectbox("Select a numeric column for values (e.g., Sales)", df.columns)  # Allow all columns

                            if group_col and value_col:
                                # Check if value_col is numeric, if not, try to convert
                                try:
                                    # Convert value_col to numeric, coerce errors to NaN
                                    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                                    if df[value_col].isna().all():
                                        st.error(f"Column '{value_col}' contains non-numeric values that cannot be converted. Please select a numeric column.")
                                        return

                                    # Pivot the data for grouped bar chart
                                    pivot_df = df.pivot_table(index=group_col, values=value_col, aggfunc='mean').fillna(0)

                                    if visualization_type == "Matplotlib":
                                        plt.figure(figsize=(10,5), facecolor='#1C2526')
                                        pivot_df.plot(kind='bar', ax=plt.gca(), color=['#FF4040', '#FFFF40', '#40FF40', '#4040FF'])
                                        plt.xlabel(group_col)
                                        plt.ylabel(value_col)
                                        plt.title("Grouped Bar Chart")
                                        plt.gcf().set_facecolor('#1C2526')
                                        st.pyplot(plt)
                                        plt.clf()

                                    elif visualization_type == "Seaborn":
                                        plt.figure(figsize=(10,5), facecolor='#1C2526')
                                        sns.barplot(data=df, x=value_col, y=group_col, hue=group_col, palette='Set1')
                                        plt.gcf().set_facecolor('#1C2526')
                                        st.pyplot(plt)
                                        plt.clf()

                                    else:  # Plotly
                                        fig = px.bar(df, x=value_col, y=group_col, color=group_col, 
                                                     color_discrete_sequence=px.colors.qualitative.Set1,
                                                     title="Grouped Bar Chart")
                                        st.plotly_chart(fig)
                                except Exception as e:
                                    st.error(f"Error creating bar plot: {e}")
                                    return
                            else:
                                st.error("Please select both a categorical column and a numeric column.")
                        else:
                            st.error("No categorical columns available for grouping.")

                    # Scatter Plot
                    if st.checkbox("Show Scatter Plot"):
                        if visualization_type == "Plotly":
                            fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1] if len(selected_columns) > 1 else None, color=selected_columns[0], color_continuous_scale='Viridis')
                            st.plotly_chart(fig)
                        else:
                            st.scatter_chart(df[selected_columns])

                    # Histogram
                    if st.checkbox("Show Histogram"):
                        if visualization_type == "Plotly":
                            fig = px.histogram(df, x=selected_columns[0], color_discrete_sequence=px.colors.sequential.Rainbow)
                            st.plotly_chart(fig)
                        else:
                            df[selected_columns].hist(figsize=(10,5), color='orange')
                            plt.gcf().set_facecolor('#1C2526')
                            st.pyplot(plt)
                            plt.clf()

                    # Line Plot
                    if st.checkbox("Show Line Plot"):
                        if visualization_type == "Plotly":
                            fig = px.line(df, x=selected_columns[0], y=selected_columns[1] if len(selected_columns) > 1 else None, color_discrete_sequence=px.colors.sequential.Viridis)
                            st.plotly_chart(fig)
                        else:
                            st.line_chart(df[selected_columns])

                    # Pie Chart
                    if st.checkbox("Show Pie Chart"):
                        if len(selected_columns) >= 1:
                            pie_df = df[selected_columns].sum()
                            fig = px.pie(names=pie_df.index, values=pie_df.values, title="Pie Chart", color_discrete_sequence=px.colors.sequential.Rainbow)
                            st.plotly_chart(fig)
                        else:
                            st.error("Select at least one column for Pie Chart")

                    # Box Plot
                    if st.checkbox("Show Box Plot"):
                        if visualization_type == "Seaborn":
                            plt.figure(figsize=(10,5), facecolor='#1C2526')
                            sns.boxplot(data=df[selected_columns], palette='rainbow')
                            st.pyplot(plt)
                            plt.clf()
                        else:
                            fig = px.box(df, y=selected_columns, color_discrete_sequence=px.colors.sequential.Viridis)
                            st.plotly_chart(fig)

                    # Elbow Plot
                    if st.checkbox("Show Elbow Plot"):
                        distortions = []
                        for i in range(1, 11):
                            kmeans = KMeans(n_clusters=i)
                            kmeans.fit(df[selected_columns])
                            distortions.append(kmeans.inertia_)
                        plt.figure(figsize=(8,5), facecolor='#1C2526')
                        plt.plot(range(1, 11), distortions, marker='o', color='cyan')
                        plt.xlabel('Number of clusters')
                        plt.ylabel('Distortion')
                        st.pyplot(plt)
                        plt.clf()

                    # Violin Plot
                    if st.checkbox("Show Violin Plot"):
                        if visualization_type == "Seaborn":
                            plt.figure(figsize=(10,5), facecolor='#1C2526')
                            sns.violinplot(data=df[selected_columns], palette='rainbow')
                            st.pyplot(plt)
                            plt.clf()
                        else:
                            fig = px.violin(df, y=selected_columns[0], box=True, points="all", color_discrete_sequence=px.colors.sequential.Viridis)
                            st.plotly_chart(fig)

                    # Pair Plot
                    if st.checkbox("Show Pair Plot"):
                        if visualization_type == "Seaborn":
                            pair_plot = sns.pairplot(df[selected_columns], palette='rainbow')
                            pair_plot.fig.set_facecolor('#1C2526')
                            st.pyplot(pair_plot.fig)
                            plt.clf()
                        else:
                            st.write("Pair Plot sirf Seaborn me available hai.")

                    # 3D Scatter Plot
                    if st.checkbox("Show 3D Scatter Plot") and len(selected_columns) >= 3:
                        if visualization_type == "Plotly":
                            fig = px.scatter_3d(df, x=selected_columns[0], y=selected_columns[1], z=selected_columns[2], color=selected_columns[0], color_continuous_scale='Rainbow')
                            st.plotly_chart(fig)
                        else:
                            st.write("3D Scatter sirf Plotly me available hai.")

                    # Density Plot
                    if st.checkbox("Show Density Plot"):
                        if visualization_type == "Seaborn":
                            plt.figure(figsize=(10,5), facecolor='#1C2526')
                            for col in selected_columns:
                                sns.kdeplot(df[col],  label=col, color='red')
                            plt.legend()
                            st.pyplot(plt)
                            plt.clf()
                        else:
                            fig = px.density_contour(df, x=selected_columns[0], y=selected_columns[1] if len(selected_columns) > 1 else None, color_discrete_sequence=px.colors.sequential.Viridis)
                            st.plotly_chart(fig)

            if "Classification" in tasks or "Regression" in tasks:
                if len(df.columns) > 0:
                    target_column = st.selectbox("Select Target Column", df.columns, key="target_column_2")
                    load_model = st.checkbox("Load Saved Model")
                    
                    if load_model:
                        model_file = st.file_uploader("Upload saved model (.joblib)", type=["joblib"])
                        if model_file:
                            best_model = joblib.load(model_file)
                            st.write(f"Loaded Model: {best_model}")
                            best_score = "N/A (Loaded Model)"
                    elif st.button("Train Model"):
                        if target_column:
                            import pickle
                            import datetime

                            with st.spinner("Training model..."):
                                try:
                                    best_model, best_score = train_model(df, target_column)  # problem_type parameter hata diya
                                except ValueError as e:
                                    st.error(f"Training Error: {e}")
                                    return

                            # Show on screen
                            st.write(f"✅ Best Model: {best_model}")
                            st.write(f"🎯 Best Score: {best_score:.4f}")

                            # Save files with timestamp
                            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            model_filename = f"trained_model_{now}.pkl"
                            report_filename = f"training_report_{now}.txt"

                            # Save model
                            with open(model_filename, "wb") as f:
                                pickle.dump(best_model, f)

                            # Save report
                            with open(report_filename, "w", encoding="utf-8") as f:
                                f.write("📊 ML Training Report\n")
                                f.write(f"Timestamp: {now}\n")
                                f.write(f"Target Column: {target_column}\n")
                                # f.write(f"Problem Type: {problem_type}\n")  # Yeh line hata di kyunki problem_type ab manually nahi decide hota
                                f.write(f"Best Model: {best_model}\n")
                                f.write(f"Best Score: {best_score:.4f}\n")

                            st.success("✅ Model and report saved successfully.")

                            # Download buttons
                            with open(model_filename, "rb") as f:
                                st.download_button("📥 Download Trained Model (.pkl)", f, file_name=model_filename)

                            with open(report_filename, "rb") as f:
                                st.download_button("📄 Download Training Report (.txt)", f, file_name=report_filename)

                        else:
                            st.error("⚠️ Please select a valid target column.")
                else:
                    st.error("⚠️ No valid columns available for target selection.")
            
            if "Clustering" in tasks:
                import datetime
                import pandas as pd

                # Clustering UI aur computation
                clusters = st.slider("Select number of clusters (if applicable)", 2, 10, 3)
                clustering_method, model, clusters, show_elbow, show_viz = clustering_ui(df, clusters)
                
                with st.spinner("Performing clustering..."):
                    try:
                        df, numeric_df = perform_clustering(df, model)
                    except ValueError as e:
                        st.error(f"Clustering Error: {e}")
                        return

                # Elbow plot for KMeans
                if show_elbow:
                    distortions = []
                    for i in range(1, 11):
                        kmeans = KMeans(n_clusters=i)
                        kmeans.fit(numeric_df)
                        distortions.append(kmeans.inertia_)
                    fig, ax = plt.subplots()
                    ax.plot(range(1, 11), distortions, marker='o')
                    ax.set_xlabel('Number of clusters')
                    ax.set_ylabel('Distortion')
                    ax.set_title('Elbow Plot')
                    st.pyplot(fig)
                    plt.clf()

                # Visualization
                if show_viz:
                    fig = px.scatter(df, x=numeric_df.columns[0], y=numeric_df.columns[1] if len(numeric_df.columns) > 1 else numeric_df.columns[0], 
                                     color='Cluster', title="Cluster Visualization")
                    st.plotly_chart(fig)

                st.write("✅ Clustered Data")
                st.dataframe(df.head())

                # Save Report
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cluster_report_file = f"clustering_report_{now}.txt"
                clustered_data_file = f"clustered_data_{now}.csv"

                with open(cluster_report_file, "w", encoding="utf-8") as f:
                    f.write("📊 Clustering Report\n")
                    f.write(f"Timestamp: {now}\n")
                    f.write(f"Number of Clusters: {clusters}\n")
                    f.write(f"Columns used: {', '.join(df.columns)}\n")
                    f.write("Clustering performed successfully.\n")

                df.to_csv(clustered_data_file, index=False)

                # Download Buttons
                with open(cluster_report_file, "rb") as f:
                    st.download_button("📄 Download Clustering Report (.txt)", f, file_name=cluster_report_file)

                with open(clustered_data_file, "rb") as f:
                    st.download_button("📥 Download Clustered Data (.csv)", f, file_name=clustered_data_file)

            if "Time Series Analysis" in tasks:
                import datetime

                column = st.selectbox("Select column for time series analysis", df.columns, key="time_series_column")
                if st.button("Run Time Series Analysis"):
                    with st.spinner("Analyzing time series..."):
                        forecast = time_series_analysis(df, column)

                    if forecast is not None:
                        st.write("📈 Time Series Forecast:")
                        st.line_chart(forecast)

                        # Save report
                        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        ts_report_file = f"time_series_report_{now}.txt"
                        forecast_file = f"forecast_data_{now}.csv"

                        with open(ts_report_file, "w", encoding="utf-8") as f:
                            f.write("📊 Time Series Analysis Report\n")
                            f.write(f"Timestamp: {now}\n")
                            f.write(f"Forecasted Column: {column}\n")
                            f.write("Forecast completed successfully.\n")

                        forecast.to_csv(forecast_file)

                        # Download buttons
                        with open(ts_report_file, "rb") as f:
                            st.download_button("📄 Download Time Series Report (.txt)", f, file_name=ts_report_file)

                        with open(forecast_file, "rb") as f:
                            st.download_button("📥 Download Forecast Data (.csv)", f, file_name=forecast_file)

    
    # Footer Section
    st.markdown("""
        <div class="footer">
            <h3>Contact Me</h3>
            <p>Name: Mohid Khan</p>
            <p>Phone: +92 333 0215061</p>
            <p>Email: <a href="mailto:Mohidadil24@gmail.com">Mohidadil24@gmail.com</a></p>
            <p>
                <a href="https://github.com/mohidadil" target="_blank">GitHub</a> |
                <a href="https://www.linkedin.com/in/mohid-adil-101b5b295/" target="_blank">LinkedIn</a>
            </p>
            <h4>Quick Links</h4>
            <p>
                <a href="#home">Home</a> |
                <a href="#about">About</a> |
                <a href="#contact">Contact Us</a>
            </p>
            <p>© 2025 Your Name. All Rights Reserved.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()