import os
import pathlib
import dataclasses
import numpy
import matplotlib.pyplot
import sklearn.cluster
import sklearn.datasets

SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_PATH.parent
BUILD_DIR = SCRIPT_DIR / 'build'

@dataclasses.dataclass
class MLData:
    features: numpy.ndarray
    true_labels: numpy.ndarray

@dataclasses.dataclass
class KMeansResult:
    cluster_labels: numpy.ndarray
    cluster_centers: numpy.ndarray

def generate_dataset(samples: int, centers: int) -> MLData:
    features, labels = sklearn.datasets.make_blobs(
        n_samples=samples, 
        centers=centers, 
        cluster_std=0.8, 
        random_state=42
    )
    return MLData(features=features, true_labels=labels)

def plot_elbow_curve(data: MLData, max_k: int) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    wcss: list[float] = []
    
    for i in range(1, max_k + 1):
        kmeans = sklearn.cluster.KMeans(
            n_clusters=i, 
            init='k-means++', 
            max_iter=300, 
            n_init=10, 
            random_state=42
        )
        kmeans.fit(data.features)
        wcss.append(float(kmeans.inertia_))
        
    fig = matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-', color='#4b5de4')
    matplotlib.pyplot.title('Метод ліктя для визначення оптимальної кількості кластерів')
    matplotlib.pyplot.xlabel('Кількість кластерів (K)')
    matplotlib.pyplot.ylabel('Сума квадратів відстаней (WCSS)')
    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.7)
    
    output_filepath = BUILD_DIR / 'elbow_method.svg'
    matplotlib.pyplot.savefig(output_filepath, bbox_inches='tight', dpi=300)
    matplotlib.pyplot.close(fig)

def apply_kmeans(data: MLData, optimal_k: int) -> KMeansResult:
    kmeans = sklearn.cluster.KMeans(
        n_clusters=optimal_k, 
        init='k-means++', 
        max_iter=300, 
        n_init=10, 
        random_state=42
    )
    predicted_labels = kmeans.fit_predict(data.features)
    
    return KMeansResult(
        cluster_labels=predicted_labels, 
        cluster_centers=kmeans.cluster_centers_
    )

def plot_clusters(data: MLData, result: KMeansResult) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    fig = matplotlib.pyplot.figure(figsize=(10, 6))
    
    unique_labels = numpy.unique(result.cluster_labels)
    colors = ['#4bb2c5', '#EAA228', '#579575', '#953579', '#839557', '#c5b47f']
    
    for label in unique_labels:
        cluster_points = data.features[result.cluster_labels == label]
        matplotlib.pyplot.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            s=50, 
            color=colors[label % len(colors)], 
            label=f'Кластер {label + 1}',
            alpha=0.7
        )
        
    matplotlib.pyplot.scatter(
        result.cluster_centers[:, 0], 
        result.cluster_centers[:, 1], 
        s=200, 
        c='red', 
        marker='X', 
        label='Центроїди'
    )
    
    matplotlib.pyplot.title('Результат кластеризації методом K-Means')
    matplotlib.pyplot.xlabel('Ознака 1')
    matplotlib.pyplot.ylabel('Ознака 2')
    matplotlib.pyplot.legend(loc='upper right')
    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.5)
    
    output_filepath = BUILD_DIR / 'kmeans_clusters.svg'
    matplotlib.pyplot.savefig(output_filepath, bbox_inches='tight', dpi=300)
    matplotlib.pyplot.close(fig)

def main() -> None:
    dataset = generate_dataset(samples=400, centers=4)
    
    plot_elbow_curve(dataset, max_k=10)
    
    kmeans_result = apply_kmeans(dataset, optimal_k=4)
    
    plot_clusters(dataset, kmeans_result)

if __name__ == '__main__':
    main()