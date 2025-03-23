# Data-pipeline-development
def preprocess_data(self):
    """Preprocess data by handling missing values and scaling"""
    try:
        # Define numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()

        # Pipelines for transformations
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))  # Fixed sparse_output issue
        ])

        # Combine transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        # Apply transformations
        self.data_processed = preprocessor.fit_transform(self.data)

        # Retrieve feature names safely
        if hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
            cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
        else:
            cat_feature_names = [f"{col}_{i}" for col in categorical_cols for i in range(len(set(self.data[col].dropna())))]

        all_feature_names = numerical_cols + cat_feature_names.tolist()

        # Convert to DataFrame
        self.data_processed = pd.DataFrame(self.data_processed, columns=all_feature_names)

        print("Data preprocessing complete.")
    except Exception as e:
        print(f"Error preprocessing data: {e}")

def load_data(self):
    """Save processed data to a new CSV"""
    try:
        # Ensure self.data_processed is a DataFrame with column names
        if isinstance(self.data_processed, np.ndarray):
            self.data_processed = pd.DataFrame(self.data_processed, columns=all_feature_names)

        self.data_processed.to_csv(self.output_path, index=False)
        print(f"Processed data saved to {self.output_path}.")
    except Exception as e:
        print(f"Error loading data: {e}")
