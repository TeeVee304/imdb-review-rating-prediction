import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, silhouette_score

# 1
def class_stats(target, predicted, title=""):
    """
    Imprime as métricas de classificação: accuracy, precision, recall e matriz de confusão.
    Args:
    - target: classes verdadeiras
    - predicted: classes previstas
    - title: título para a saída (opcional)
    """
    cm = confusion_matrix(target, predicted)
    accuracy = accuracy_score(target, predicted)
    precision = precision_score(target, predicted,average="macro")
    recall = recall_score(target, predicted,average="macro")

    print(title)
    print("Accuracy:", np.round(accuracy*100,2),"%")
    print("Precision:", np.round(precision*100,2),"%")
    print("Recall:", np.round(recall*100,2),"%")
    print(f"Matriz de confusão : \n",cm)

# 2    
def reg_stats(target, predicted, title=""):
    """
    Avalia um modelo de regressão e converte as previsões para classes 
    para permitir comparação com modelos de classificação.
    Args:
    - target: pontuações verdadeiras (classes 1-4, 7-10)
    - predicted: valores previstos pelo regressor (contínuos)
    - title: título para a saída (opcional)
    """
    mae = mean_absolute_error(target, predicted)
    mse = mean_squared_error(target, predicted)
    r2 = r2_score(target, predicted)
    
    y_aux = np.round(predicted)
    y_aux[y_aux > 10.] = 10.
    y_aux[y_aux < 1.] = 1.
    y_aux[y_aux == 5.] = 4.
    y_aux[y_aux == 6.] = 7.
    cm = confusion_matrix(target, y_aux)
    accuracy = accuracy_score(target, y_aux)
    
    print(title)
    print("erro quadrático absoluto :",np.round(mae,2))
    print("erro quadratico medio :",np.round(mse,2))
    print("R2 :",np.round(r2,2))
    print("Accuracy:", np.round(accuracy*100,2),"%")
    print("Matriz de confusão :\n",cm)
    
# 3
def cluster_top_words(vectorizer, kmeans, word_count=10):
    """
    Imprime os termos mais representativos de cada cluster baseando-se nos 
    pesos dos centróides. Ajuda a identificar o tópico semântico do grupo.
    Args:
    - vectorizer: Objeto TfidfVectorizer ajustado.
    - kmeans: Modelo de clustering (KMeans).
    - word_count: Número de palavras de topo a exibir por cluster (default: 10).
    """
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    for i in range(len(order_centroids)):
        print(f"\n\nCluster {i+1} top {word_count} words:")
        for ind in order_centroids[i, :word_count]:
            print(terms[ind], end=" | ")

# 4            
def cluster_metrics(data, clusters, sample_size=2000):
    """
    Avalia a qualidade do cluster calculando o Silhouette Score (via amostragem) 
    e visualiza a distribuição das críticas por classe.
    Args:
    - data: Matriz de características (X) utilizada para treinar o modelo.
    - clusters: Array com as classes (labels dos clusters) atribuídas a cada documento.
    - sample_size: Quantidade de exemplos a usar para estimar o Silhouette Score (default: 2000).
    """
    if data.shape[0] > sample_size:
        idx = np.random.choice(data.shape[0], sample_size, replace=False)
        sil = silhouette_score(data[idx], clusters[idx])
    else:
        sil = silhouette_score(data, clusters)

    unique, counts = np.unique(clusters, return_counts=True)

    print("\nSilhouette score (amostrado):", sil, "\n")
    for i in range(len(unique)):
        print(f"Cluster {unique[i]+1}  nº críticas: {counts[i]}")

    plt.figure(figsize=(12, 3))
    plt.bar(unique+1, counts)
    plt.xlabel("Cluster")
    plt.ylabel("Número de críticas")
    plt.title("Distribuição das críticas por cluster")
    plt.xticks(unique+1)
    plt.show()