import os
import pathlib
import pandas
import numpy
import matplotlib.pyplot
import sklearn.neural_network
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.decomposition
import dataclasses

SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_PATH.parent
BUILD_DIR = SCRIPT_DIR / 'build'

@dataclasses.dataclass
class ProcessedData:
    features: pandas.DataFrame
    target: pandas.Series

def setup_build_dir() -> None:
    os.makedirs(BUILD_DIR, exist_ok=True)
    assert os.path.exists(BUILD_DIR)

def load_data(sample_path: pathlib.Path, desc_path: pathlib.Path) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    assert sample_path.exists()
    assert desc_path.exists()
    sample_df = pandas.read_csv(sample_path)
    desc_df = pandas.read_csv(desc_path)
    return sample_df, desc_df

def clean_and_prepare_data(sample_df: pandas.DataFrame, desc_df: pandas.DataFrame) -> ProcessedData:
    valid_fields = desc_df['Field_in_data'].dropna().tolist()
    existing_columns = [col for col in valid_fields if col in sample_df.columns]
    
    filtered_df = sample_df[existing_columns].copy()
    
    assert 'loan_overdue' in sample_df.columns
    target_series = sample_df['loan_overdue'].apply(lambda x: 1 if float(x) == 0 else 0)
    
    if 'loan_overdue' in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=['loan_overdue'])
        
    filtered_df = filtered_df.replace('NULL', pandas.NA)
    numeric_df = filtered_df.apply(pandas.to_numeric, errors='coerce')
    
    numeric_df = numeric_df.dropna(axis=1, thresh=int(len(numeric_df) * 0.5))
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    assert len(numeric_df.columns) > 0
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaled_features = pandas.DataFrame(
        scaler.fit_transform(numeric_df),
        columns=numeric_df.columns,
        index=numeric_df.index
    )
    
    return ProcessedData(features=scaled_features, target=target_series)

def train_neural_network(data: ProcessedData) -> numpy.ndarray:
    mlp = sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=2000,
        random_state=42
    )
    mlp.fit(data.features, data.target)
    predictions = mlp.predict(data.features)
    
    pca = sklearn.decomposition.PCA(n_components=2)
    reduced_features = pca.fit_transform(data.features)
    
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=predictions,
        cmap='coolwarm',
        alpha=0.7
    )
    matplotlib.pyplot.title('Neural Network Classification')
    matplotlib.pyplot.xlabel('PCA Component 1')
    matplotlib.pyplot.ylabel('PCA Component 2')
    
    output_path = BUILD_DIR / 'nn_classification.svg'
    matplotlib.pyplot.savefig(output_path, format='svg')
    matplotlib.pyplot.close()
    assert os.path.exists(output_path)
    
    return predictions

def detect_fraud(features: pandas.DataFrame) -> numpy.ndarray:
    isolation_forest = sklearn.ensemble.IsolationForest(
        contamination=0.1,
        random_state=42
    )
    fraud_predictions = isolation_forest.fit_predict(features)
    
    pca = sklearn.decomposition.PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=fraud_predictions,
        cmap='Set1',
        alpha=0.7
    )
    matplotlib.pyplot.title('Fraud Detection')
    matplotlib.pyplot.xlabel('PCA Component 1')
    matplotlib.pyplot.ylabel('PCA Component 2')
    
    output_path = BUILD_DIR / 'fraud_detection.svg'
    matplotlib.pyplot.savefig(output_path, format='svg')
    matplotlib.pyplot.close()
    assert os.path.exists(output_path)
    
    return fraud_predictions

def save_results(data: ProcessedData, nn_preds: numpy.ndarray, fraud_preds: numpy.ndarray) -> None:
    results_df = pandas.DataFrame(index=data.features.index)
    results_df['Will_Return_Credit'] = nn_preds
    results_df['Is_Fraud'] = [1 if x == -1 else 0 for x in fraud_preds]
    
    output_path = BUILD_DIR / 'scoring_results.csv'
    results_df.to_csv(output_path)
    assert os.path.exists(output_path)

def main() -> None:
    setup_build_dir()
    
    sample_data_path = SCRIPT_DIR / 'Lab_work_8' / 'sample_data.csv'
    desc_data_path = SCRIPT_DIR / 'Lab_work_8' / 'data_description.csv'
    
    sample_df, desc_df = load_data(sample_data_path, desc_data_path)
    processed_data = clean_and_prepare_data(sample_df, desc_df)
    
    assert len(processed_data.features) > 0
    assert len(processed_data.target) > 0
    
    nn_predictions = train_neural_network(processed_data)
    fraud_predictions = detect_fraud(processed_data.features)
    
    save_results(processed_data, nn_predictions, fraud_predictions)

if __name__ == '__main__':
    main()